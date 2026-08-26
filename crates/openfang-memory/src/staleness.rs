//! ANAI-259: how long a tier-3 claim is believed before it should be doubted.
//!
//! # The clock is not `last_read_at`
//!
//! Reading a fact does not make it truer. `deploy.live_binary` was read
//! repeatedly while it named a binary three rebuilds old; every one of those
//! reads spread the rot rather than curing it. The only event that says
//! anything about a claim's truth is a *write* — a create, an affirmation, or
//! a supersession — because each of those is a moment somebody checked the
//! claim against the world.
//!
//! That clock already exists and needed no column: [`crate::fact::FactStore`]
//! stamps `created_at` on create and on supersession, and moves
//! `last_affirmed_at` on affirmation. So `max(created_at, last_affirmed_at)`
//! *is* "last verified at", and reads never touch either. What was missing was
//! not a timestamp but the answer to "how long is too long", which is not a
//! property of the clock. It is a property of the claim.
//!
//! # Different claims rot at different speeds
//!
//! Ben's framing, which is the one to keep: meet someone at a conference and
//! their *name* is permanent, their *job* is good for months, their *opinion
//! of the talk* for days, and *who they were just talking to* for minutes. One
//! global TTL over that set is wrong four ways at once. So each slot declares
//! a [`PersistenceClass`] and the durations live in fleet config, symbolic
//! rather than a per-row TTL — the same reasoning as `working_set_ratio`:
//! retune "how long is stable" in one place instead of rewriting stored rows.
//!
//! # Past doubt is advisory, never withheld
//!
//! A stale claim is surfaced *with its age*, not suppressed. Withholding is
//! the safer-looking option and the worse one: an agent that cannot see the
//! slot cannot re-verify it, and silence at rehydration time reads as "nothing
//! is known" rather than "this needs checking". Ben's test case is the whole
//! specification — *"Hey Jim! You still pushing paper down at P&G?"* The claim
//! is what makes the question askable; the marker is what makes it a question
//! instead of an assertion.

use std::sync::atomic::{AtomicU64, Ordering};

/// How fast a claim is expected to rot.
///
/// Stored as text on the slot row, symbolic rather than a duration. The class
/// is a semantic judgment about the *kind* of claim — the sort of call a model
/// makes well — while the durations it maps to are an operator tuning knob.
/// Conflating them would put "is 90 days right for `stable`?" into every write
/// site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PersistenceClass {
    /// Never doubted. Identity, naming, physical law: "the table is
    /// `memories`, not `memory_corpus`".
    Permanent,
    /// Doubted after months. Architecture and process: the trunk model, the
    /// Linear team a lane lives in.
    Stable,
    /// Doubted after about a week. The default, and deliberately so — see
    /// [`PersistenceClass::from_stored`].
    #[default]
    Active,
    /// Doubted within a day. Deployment state, `HEAD`, anything that a rebuild
    /// invalidates. Legal, but a smell: see [`PersistenceClass::is_smell`].
    Volatile,
}

impl PersistenceClass {
    /// The stored form. This is what lands in `memories.persistence_class`.
    pub fn as_str(self) -> &'static str {
        match self {
            PersistenceClass::Permanent => "permanent",
            PersistenceClass::Stable => "stable",
            PersistenceClass::Active => "active",
            PersistenceClass::Volatile => "volatile",
        }
    }

    /// Parse a caller-supplied value, quoting the vocabulary back on a miss.
    ///
    /// Strict, because this comes from a tool argument where a typo should be
    /// corrected rather than silently absorbed into the default.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "permanent" => Ok(PersistenceClass::Permanent),
            "stable" => Ok(PersistenceClass::Stable),
            "active" => Ok(PersistenceClass::Active),
            "volatile" => Ok(PersistenceClass::Volatile),
            other => Err(format!(
                "unknown persistence class {other:?} (expected one of: permanent, stable, \
                 active, volatile)"
            )),
        }
    }

    /// Read a stored value, falling back to [`PersistenceClass::Active`].
    ///
    /// Unlike [`Self::parse`] this cannot fail, and the fallback is `Active`
    /// rather than `Permanent` on purpose. Every row written before v16 has a
    /// NULL here, and the failure modes are not symmetric: over-flagging is
    /// noise a reader ignores, under-flagging is a confident lie told at
    /// rehydration time, when a fact carries *more* authority than usual
    /// because it is one of the few things the agent has. Unclassified is the
    /// dangerous state, so it gets the middling clock, not the silent one.
    pub fn from_stored(s: Option<&str>) -> Self {
        s.and_then(|s| Self::parse(s).ok()).unwrap_or_default()
    }

    /// True for a class that says the slot store is being used as a
    /// scratchpad.
    ///
    /// A claim that rots in hours is usually episode material wearing a slot.
    /// It stays legal — `repo.trunk_head` is the honest exception, genuinely
    /// volatile and genuinely worth a slot — but the writer logs it, so the
    /// pattern is visible before it is a habit.
    pub fn is_smell(self) -> bool {
        matches!(self, PersistenceClass::Volatile)
    }

    /// How long this class is believed, under `policy`. `None` is "forever".
    pub fn doubt_after_days(self, policy: StalenessPolicy) -> Option<f64> {
        match self {
            PersistenceClass::Permanent => None,
            PersistenceClass::Stable => Some(policy.stable_days),
            PersistenceClass::Active => Some(policy.active_days),
            PersistenceClass::Volatile => Some(policy.volatile_days),
        }
    }
}

impl std::fmt::Display for PersistenceClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Compiled defaults, in days. Overridable by fleet config, one place.
pub const DEFAULT_STABLE_DAYS: f64 = 90.0;
/// See [`DEFAULT_STABLE_DAYS`].
pub const DEFAULT_ACTIVE_DAYS: f64 = 7.0;
/// See [`DEFAULT_STABLE_DAYS`].
pub const DEFAULT_VOLATILE_DAYS: f64 = 1.0;

/// The class-to-duration mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StalenessPolicy {
    /// Days before a `stable` claim should be re-verified.
    pub stable_days: f64,
    /// Days before an `active` claim should be re-verified.
    pub active_days: f64,
    /// Days before a `volatile` claim should be re-verified.
    pub volatile_days: f64,
}

impl Default for StalenessPolicy {
    fn default() -> Self {
        Self {
            stable_days: DEFAULT_STABLE_DAYS,
            active_days: DEFAULT_ACTIVE_DAYS,
            volatile_days: DEFAULT_VOLATILE_DAYS,
        }
    }
}

/// Reject a policy that is not a ladder.
///
/// The classes are ordered by construction — `volatile` rots faster than
/// `active` rots faster than `stable` — and an inverted config would not fail
/// loudly, it would quietly make `stable` the twitchiest class in the system.
/// Pure, so the rules are testable without touching the process globals every
/// other test in the binary reads through.
pub fn validate_policy(p: StalenessPolicy) -> Result<(), String> {
    for (name, v) in [
        ("volatile_days", p.volatile_days),
        ("active_days", p.active_days),
        ("stable_days", p.stable_days),
    ] {
        if !v.is_finite() || v <= 0.0 {
            return Err(format!(
                "fact staleness {name} must be a positive, finite number of days (got {v})"
            ));
        }
    }
    if !(p.volatile_days <= p.active_days && p.active_days <= p.stable_days) {
        return Err(format!(
            "fact staleness durations must be a ladder: volatile_days ({}) <= active_days \
             ({}) <= stable_days ({})",
            p.volatile_days, p.active_days, p.stable_days
        ));
    }
    Ok(())
}

// `0` is the nothing-installed sentinel: `0.0f64` has an all-zero bit pattern
// and is refused by `validate_policy` anyway, so it can never be a real value.
static STABLE_DAYS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_DAYS: AtomicU64 = AtomicU64::new(0);
static VOLATILE_DAYS: AtomicU64 = AtomicU64::new(0);

/// Install the fleet policy at boot. Refuses the value, not the boot.
///
/// Same trade as `install_working_set_ratio` (ANAI-260): a mistyped tuning
/// number logs an error and leaves the compiled defaults live, rather than
/// taking a hundred agents offline at 3am over a config typo.
pub fn install_policy(p: StalenessPolicy) -> Result<(), String> {
    validate_policy(p)?;
    STABLE_DAYS.store(p.stable_days.to_bits(), Ordering::Relaxed);
    ACTIVE_DAYS.store(p.active_days.to_bits(), Ordering::Relaxed);
    VOLATILE_DAYS.store(p.volatile_days.to_bits(), Ordering::Relaxed);
    Ok(())
}

fn load(cell: &AtomicU64, fallback: f64) -> f64 {
    match cell.load(Ordering::Relaxed) {
        0 => fallback,
        bits => f64::from_bits(bits),
    }
}

/// The live policy: whatever was installed, else the compiled defaults.
pub fn policy() -> StalenessPolicy {
    StalenessPolicy {
        stable_days: load(&STABLE_DAYS, DEFAULT_STABLE_DAYS),
        active_days: load(&ACTIVE_DAYS, DEFAULT_ACTIVE_DAYS),
        volatile_days: load(&VOLATILE_DAYS, DEFAULT_VOLATILE_DAYS),
    }
}

/// What a reader should be told about a claim's age.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Staleness {
    /// Inside its class's window, or `permanent`. Say nothing.
    Fresh,
    /// Past its window. Surface the claim *and* this, never one without the
    /// other.
    ShouldVerify {
        /// Whole days since the claim was last verified.
        age_days: i64,
    },
}

impl Staleness {
    /// True when the reader should be warned.
    pub fn should_verify(self) -> bool {
        matches!(self, Staleness::ShouldVerify { .. })
    }
}

/// Judge one claim's age against its class.
///
/// `age_days` may be negative if a row is stamped in the future (clock skew,
/// a hand-edited row); that is never stale, and reporting it as such would be
/// a confusing lie about a row that is, if anything, too fresh.
pub fn judge(class: PersistenceClass, age_days: f64, policy: StalenessPolicy) -> Staleness {
    let Some(limit) = class.doubt_after_days(policy) else {
        return Staleness::Fresh;
    };
    if age_days.is_finite() && age_days > limit {
        Staleness::ShouldVerify {
            age_days: age_days.floor() as i64,
        }
    } else {
        Staleness::Fresh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unclassified row — every row written before v16 — must land on the
    /// middling clock, not the silent one. Under-flagging is the asymmetric
    /// failure: it is a confident lie at rehydration time.
    #[test]
    fn unclassified_defaults_to_active_not_permanent() {
        assert_eq!(
            PersistenceClass::from_stored(None),
            PersistenceClass::Active
        );
        assert_eq!(
            PersistenceClass::from_stored(Some("nonsense")),
            PersistenceClass::Active
        );
        assert!(PersistenceClass::default()
            .doubt_after_days(StalenessPolicy::default())
            .is_some());
    }

    /// A tool argument typo should be corrected, not absorbed.
    #[test]
    fn parse_is_strict_where_from_stored_is_lenient() {
        assert_eq!(
            PersistenceClass::parse("Stable").unwrap(),
            PersistenceClass::Stable
        );
        let err = PersistenceClass::parse("semi-permanent").unwrap_err();
        assert!(err.contains("permanent"), "the error quotes the vocabulary");
    }

    /// `permanent` is the one class with no clock. Names do not go stale.
    #[test]
    fn permanent_is_never_stale_however_old() {
        assert_eq!(
            judge(
                PersistenceClass::Permanent,
                100_000.0,
                StalenessPolicy::default()
            ),
            Staleness::Fresh
        );
    }

    /// Ben's Jim case: a stable claim last verified well over the window
    /// surfaces with its age, so the agent asks rather than asserts.
    #[test]
    fn a_stable_claim_past_its_window_asks_to_be_verified() {
        assert_eq!(
            judge(PersistenceClass::Stable, 104.5, StalenessPolicy::default()),
            Staleness::ShouldVerify { age_days: 104 }
        );
        assert_eq!(
            judge(PersistenceClass::Stable, 89.0, StalenessPolicy::default()),
            Staleness::Fresh
        );
    }

    /// The classes must stay ordered: the same age judged against a faster
    /// class is stale sooner, never later.
    #[test]
    fn faster_classes_go_stale_sooner() {
        let p = StalenessPolicy::default();
        let age = 2.0;
        assert!(judge(PersistenceClass::Volatile, age, p).should_verify());
        assert!(!judge(PersistenceClass::Active, age, p).should_verify());
        assert!(!judge(PersistenceClass::Stable, age, p).should_verify());
    }

    /// A future-stamped row is not stale. Clock skew must not manufacture a
    /// warning about a row that is, if anything, too fresh.
    #[test]
    fn a_future_timestamp_is_not_stale() {
        assert_eq!(
            judge(PersistenceClass::Volatile, -5.0, StalenessPolicy::default()),
            Staleness::Fresh
        );
    }

    /// An inverted config would quietly make `stable` the twitchiest class in
    /// the system. Refuse it.
    #[test]
    fn an_inverted_ladder_is_refused() {
        assert!(validate_policy(StalenessPolicy::default()).is_ok());
        assert!(validate_policy(StalenessPolicy {
            stable_days: 1.0,
            active_days: 7.0,
            volatile_days: 1.0,
        })
        .is_err());
        assert!(validate_policy(StalenessPolicy {
            stable_days: 90.0,
            active_days: 0.0,
            volatile_days: 1.0,
        })
        .is_err());
    }
}
