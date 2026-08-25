//! Episodes: the boundary object episodic capture groups into (ADR 0001 §2.2).
//!
//! `memories` rows were grouped by nothing but `agent_id` and timestamp, which
//! gives consolidation no defensible input set — "last N turns" cuts
//! mid-thought and re-consolidates settled work. An **episode** is a
//! half-open interval of one agent's turns: opened lazily on the first turn
//! after a gap, closed by timer, by explicit request, or (later) by an
//! agent-judged topic switch.
//!
//! # Why the DB owns the lifecycle, not the agent
//!
//! The "currently open episode" is not in-memory state on the agent handle; it
//! is *the row whose `closed_at IS NULL`*. That choice buys three things:
//!
//! 1. **Restart durability.** A daemon restart mid-conversation resumes the
//!    same episode instead of silently splitting one thought in two.
//! 2. **Structural uniqueness.** A partial unique index
//!    (`ON episodes(agent_id) WHERE closed_at IS NULL`) makes "two open
//!    episodes for one agent" a constraint violation rather than a race we
//!    hope not to lose. Same spirit as ADR §2.3.2: enforce in DDL, not wording.
//! 3. **No background task.** Close-on-timer is evaluated lazily on the next
//!    turn ([`EpisodeStore::ensure_open`]), so v1 needs no scheduler wiring.
//!
//! The cost of (3) is that an agent which goes quiet forever leaves its last
//! episode open until something touches it. [`EpisodeStore::sweep_idle`] is the
//! fleet-wide reaper for exactly that case, safe to call from the consolidation
//! tick; nothing calls it yet.
//!
//! # Staging
//!
//! v1 emits `timer` and `explicit` only. [`CloseReason::TopicSwitch`] and
//! [`CloseReason::Abandoned`] are accepted by the schema and round-trip
//! through this module so the later agent-judgment work is a caller change,
//! not a migration.

use chrono::{DateTime, Duration, Utc};
use openfang_types::agent::AgentId;
use openfang_types::error::{OpenFangError, OpenFangResult};
use rusqlite::Connection;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Default idle gap after which the next turn starts a fresh episode.
///
/// **Zero: close-on-timer is off by default.** Episodes end when they end.
///
/// The original two-hour default assumed an episode is a sitting. It is not —
/// in field use a session goes quiet for days and then resumes on exactly the
/// same topic, and a timer would have shredded that one thread into a dozen
/// fragments, each too small for consolidation to say anything useful about.
/// Wall-clock silence is not evidence that the work ended; only the agent
/// saying so is. So the sole close path is explicit `memory_episode_close`.
///
/// A positive value re-enables the timer for deployments that want it — the
/// mechanism is intact, just not armed. The cost of leaving it off is that a
/// long-lived agent's episode grows without bound if it never closes one
/// (ANAI-87's blob-growth concern, arriving through a second door). That is a
/// close-discipline problem, and the honest place to fix it is close
/// discipline, not a clock that guesses.
pub const DEFAULT_IDLE_TIMEOUT_MINUTES: i64 = 0;

/// Metadata key under which the active episode is stamped onto each captured
/// `memories` row. Also the name of the lifted column (see `semantic.rs`).
///
/// A row with **no** `episode_id` is pre-episode legacy data. There is no
/// backfill: absence is the legacy marker, which costs nothing and cannot
/// corrupt 35k existing rows by rewriting their metadata.
pub const EPISODE_ID_KEY: &str = "episode_id";

/// How far back a summariser tick is allowed to look for closed-but-unsummarised
/// episodes (ANAI-220).
///
/// The obvious selector — `closed_at IS NOT NULL AND summary IS NULL` — is not
/// a "what did this tick close" query, it is **the backfill**: on the first
/// armed tick it would sweep up every historical null-summary episode ever
/// closed. Backfill is a deliberate, separately-authorised pass, so the live
/// path is bounded by close recency instead.
///
/// It is deliberately wider than one tick. The idle sweep is not the only
/// close path: [`EpisodeStore::ensure_open`] lazily timer-closes on the
/// agent's own turn, and a 30-second model call has no business on a turn's
/// critical path — so those episodes must be picked up by a *later* tick than
/// the one that closed them.
///
/// An episode that ages out of this window keeps its null summary forever
/// (until a backfill). That is the intended cost: no answer means leave the
/// summary null and move on, never retry — a close that waits on a provider is
/// a close a provider outage can lose.
///
/// **Coupled to `episode_summary::PROBE_AFTER_TICKS`.** The breaker's half-open
/// probe waits ~30 minutes between attempts, so two failed probes span longer
/// than this window: an episode closed at the moment the breaker tripped can
/// age out before a probe ever succeeds. The probe recovers the *task*, not
/// that hour's *material*. Widening the probe interval, or narrowing this
/// window, widens that hole — change one and re-check the other.
pub const SUMMARY_LOOKBACK_MINUTES: i64 = 60;

/// Fewest episode-linked rows an episode must have before it is worth a model
/// call.
///
/// `turn_count` counts *turns*; only stored memories carry `episode_id`, and
/// measured against the live fleet the two diverge hard — 82 turns against 6
/// linked rows on one agent, 14 turns against 0 on another. Summarising an
/// episode with no material spends a call to write a polished null, so the
/// floor is on material, never on `turn_count`.
pub const MIN_MATERIAL_ROWS: usize = 2;

/// Ceiling on rows fed to one summary call. The output ceiling
/// (`consolidation.max_tokens`) bounds what comes back; this bounds what goes
/// out, so one pathological episode cannot turn a cheap call into an expensive
/// one.
pub const MAX_MATERIAL_ROWS: usize = 40;

/// `memories.kind` for a consolidation-written episode summary.
///
/// **Not `note`.** `memory_note` writes `kind = 'note'` carrying the same
/// `episode_id`, so sharing the discriminator would make "has this episode
/// already been summarised?" answer *true* for any episode where the agent
/// happened to jot something — silently skipping exactly the episodes with the
/// most material. Its own value keeps the idempotency check a single indexed
/// lookup on `episode_id` with no `json_extract` in sight.
pub const SUMMARY_KIND: &str = "summary";

/// Why an episode was closed (ADR §2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseReason {
    /// Agent-judged topic change, after human approval. Not emitted in v1.
    TopicSwitch,
    /// Someone asked for a wrap-up ("close the episode").
    Explicit,
    /// The idle gap elapsed; the next turn belongs to a new episode.
    Timer,
    /// Reaped without a clean close (crash, abandoned work). Not emitted in v1.
    Abandoned,
}

impl CloseReason {
    /// Stable wire/DB spelling.
    pub fn as_str(&self) -> &'static str {
        match self {
            CloseReason::TopicSwitch => "topic-switch",
            CloseReason::Explicit => "explicit",
            CloseReason::Timer => "timer",
            CloseReason::Abandoned => "abandoned",
        }
    }

    /// Parse the DB spelling. Unknown values are rejected rather than coerced —
    /// a reason we do not recognise is a bug upstream, not a default.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "topic-switch" => Some(CloseReason::TopicSwitch),
            "explicit" => Some(CloseReason::Explicit),
            "timer" => Some(CloseReason::Timer),
            "abandoned" => Some(CloseReason::Abandoned),
            _ => None,
        }
    }
}

impl std::fmt::Display for CloseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One episode row.
#[derive(Debug, Clone)]
pub struct Episode {
    /// Episode identity, stamped into every captured row's metadata.
    pub id: Uuid,
    /// Owning agent.
    pub agent_id: AgentId,
    /// When the first turn of this episode landed.
    pub opened_at: DateTime<Utc>,
    /// When the most recent turn of this episode landed. The timer's clock.
    pub last_activity_at: DateTime<Utc>,
    /// `None` while open.
    pub closed_at: Option<DateTime<Utc>>,
    /// Short human-legible title, written at close. `None` until then.
    pub title: Option<String>,
    /// Optional wrap-up written at close by whoever closed the episode.
    ///
    /// Distinct from the summary consolidation will later derive: this one is
    /// the closer's own account of the work, recorded at the moment the
    /// boundary is drawn. Always `None` for a timer close — nothing is present
    /// to narrate an episode that ended by going quiet, and inventing text
    /// there would put an ungrounded claim in the tier that exists to hold
    /// grounded ones.
    pub summary: Option<String>,
    /// `None` while open.
    pub close_reason: Option<CloseReason>,
    /// Turns captured into this episode.
    pub turn_count: u64,
}

impl Episode {
    /// True while `closed_at IS NULL`.
    pub fn is_open(&self) -> bool {
        self.closed_at.is_none()
    }
}

/// A point-in-time answer to "where am I in my own memory?" — the instrument
/// panel behind the `memory_status` tool (ADR 0002 §2.2).
///
/// Exists because the deferred topic-switch work (§2.6) asks an agent to notice
/// it is drifting, and an agent with no introspection is guessing. Every field
/// is derived from the episodes table at read time, so it cannot drift from the
/// lifecycle it describes.
#[derive(Debug, Clone)]
pub struct EpisodeStatus {
    /// The agent's open episode, or `None` when it has none in flight (fresh
    /// agent, or the last one closed and nothing has been captured since).
    pub current: Option<Episode>,
    /// Configured idle gap. `<= 0` means close-on-timer is disabled.
    pub idle_timeout_minutes: i64,
    /// Whole minutes since the open episode's last captured turn.
    pub idle_minutes: Option<i64>,
    /// Whole minutes before the open episode times out. `None` when nothing is
    /// open or the timer is disabled. Saturates at 0 rather than going
    /// negative: an episode past its gap is closed by the *next* turn, so 0 is
    /// the honest reading rather than -7.
    pub minutes_until_timer_close: Option<i64>,
    /// Most recently closed episodes, newest first. Bounded by the caller.
    pub recent: Vec<Episode>,
}

/// SQLite-backed episode lifecycle owner.
#[derive(Clone)]
pub struct EpisodeStore {
    conn: Arc<Mutex<Connection>>,
    idle_timeout_minutes: i64,
}

impl EpisodeStore {
    /// Create a store with the default idle timeout.
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self {
            conn,
            idle_timeout_minutes: DEFAULT_IDLE_TIMEOUT_MINUTES,
        }
    }

    /// Create a store with an explicit idle timeout (minutes).
    ///
    /// A non-positive value disables close-on-timer entirely: episodes then
    /// only ever close explicitly. That is a legitimate configuration, not an
    /// error, so it is clamped rather than rejected.
    pub fn with_idle_timeout(conn: Arc<Mutex<Connection>>, idle_timeout_minutes: i64) -> Self {
        Self {
            conn,
            idle_timeout_minutes,
        }
    }

    /// The configured idle gap in minutes; `<= 0` means timer-close is off.
    pub fn idle_timeout_minutes(&self) -> i64 {
        self.idle_timeout_minutes
    }

    /// Return the agent's open episode, if any.
    pub fn current(&self, agent_id: AgentId) -> OpenFangResult<Option<Episode>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        current_open(&conn, agent_id)
    }

    /// Fetch one episode by id, open or closed.
    pub fn get(&self, id: Uuid) -> OpenFangResult<Option<Episode>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(&format!("{SELECT_COLS} WHERE id = ?1"))
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        match stmt.query_row(rusqlite::params![id.to_string()], row_to_episode) {
            Ok(ep) => Ok(Some(ep)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(OpenFangError::Memory(e.to_string())),
        }
    }

    /// **The capture-path entry point.** Return the id of the episode this turn
    /// belongs to, opening one if needed and closing a timed-out predecessor.
    ///
    /// Exactly one of three things happens, atomically:
    ///
    /// - no open episode      -> open one, return it
    /// - open, still fresh    -> bump `last_activity_at` / `turn_count`, return it
    /// - open, but idle-timed -> close it (`timer`), open a successor, return that
    ///
    /// `turn_count` is incremented here rather than at write time because this
    /// is the single funnel both agent loops call; counting at the `remember`
    /// site would miss dropped (ANAI-76) heartbeat turns and drift.
    pub fn ensure_open(&self, agent_id: AgentId) -> OpenFangResult<Uuid> {
        let now = Utc::now();
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let existing = current_open(&tx, agent_id)?;
        let id = match existing {
            Some(ep) if !self.is_timed_out(&ep, now) => {
                tx.execute(
                    "UPDATE episodes SET last_activity_at = ?2, turn_count = turn_count + 1 \
                     WHERE id = ?1",
                    rusqlite::params![ep.id.to_string(), now.to_rfc3339()],
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                ep.id
            }
            Some(ep) => {
                // Timed out. `closed_at` is stamped at `now` — it means "when
                // the close happened", which is the only question it can answer
                // that nothing else already does.
                //
                // It used to be back-dated to `last_activity_at` on the theory
                // that the episode ended when the work did. That reading is
                // fine, but it makes the field redundant: the content span is
                // already recoverable as `opened_at..last_activity_at`, so
                // back-dating spent the one column that could tell you when the
                // lifecycle actually ran, and left closed episodes looking like
                // they closed themselves the instant they went quiet. Duration
                // of the *work* is unchanged and still exact; duration of the
                // *episode row* now includes the idle tail, which is the truth.
                //
                // No title and no summary: a timer close has no author. See
                // `Episode::summary`.
                close_row(&tx, ep.id, now, CloseReason::Timer, None, None)?;
                insert_open(&tx, agent_id, now)?
            }
            None => insert_open(&tx, agent_id, now)?,
        };

        tx.commit()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(id)
    }

    /// Close the agent's open episode. Returns the closed episode's id, or
    /// `None` when the agent had nothing open (idempotent, not an error — a
    /// double "wrap up" should be a no-op, not a failure).
    pub fn close_current(
        &self,
        agent_id: AgentId,
        reason: CloseReason,
        title: Option<&str>,
        summary: Option<&str>,
    ) -> OpenFangResult<Option<Uuid>> {
        let now = Utc::now();
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let Some(ep) = current_open(&conn, agent_id)? else {
            return Ok(None);
        };
        close_row(&conn, ep.id, now, reason, title, summary)?;
        Ok(Some(ep.id))
    }

    /// Assemble the agent's memory status. Read-only.
    ///
    /// `recent_limit` bounds the closed-episode tail. Pass 0 to skip it — the
    /// status tool wants a couple of lines of history, not a log dump.
    pub fn status(&self, agent_id: AgentId, recent_limit: usize) -> OpenFangResult<EpisodeStatus> {
        let current = self.current(agent_id)?;
        let now = Utc::now();

        let idle_minutes = current
            .as_ref()
            .map(|ep| (now - ep.last_activity_at).num_minutes().max(0));
        let minutes_until_timer_close = match (&current, self.idle_timeout_minutes) {
            (Some(_), t) if t > 0 => Some((t - idle_minutes.unwrap_or(0)).max(0)),
            _ => None,
        };

        let recent = if recent_limit == 0 {
            Vec::new()
        } else {
            // Ask for one extra and drop the open row: `list_for_agent` returns
            // newest-first including the episode in flight, and the caller
            // wants history, not the row it already has in `current`.
            self.list_for_agent(agent_id, recent_limit + 1)?
                .into_iter()
                .filter(|ep| !ep.is_open())
                .take(recent_limit)
                .collect()
        };

        Ok(EpisodeStatus {
            current,
            idle_timeout_minutes: self.idle_timeout_minutes,
            idle_minutes,
            minutes_until_timer_close,
            recent,
        })
    }

    /// Fleet-wide reaper for open episodes past the idle gap. Returns the
    /// number closed.
    ///
    /// Lazy timer-close ([`Self::ensure_open`]) only fires when an agent speaks
    /// again; this is the sweep for agents that went quiet. Safe to call from
    /// the consolidation tick.
    ///
    /// ANAI-219 gave it a caller: the kernel's episode-sweep task, on its own
    /// 60s tick, spawned only when `episode_idle_timeout_minutes > 0`. With
    /// the idle timer off by default ([`DEFAULT_IDLE_TIMEOUT_MINUTES`]) that
    /// task never spawns and this returns 0 immediately, so the default
    /// deployment is unchanged. If you are here because episodes are staying
    /// open, that is the timer being off — see the const.
    pub fn sweep_idle(&self) -> OpenFangResult<usize> {
        if self.idle_timeout_minutes <= 0 {
            return Ok(0);
        }
        let cutoff = Utc::now() - Duration::minutes(self.idle_timeout_minutes);
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        // Same "closed_at is when the close ran" rule as the lazy path.
        let now = Utc::now().to_rfc3339();
        let n = conn
            .execute(
                "UPDATE episodes SET closed_at = ?3, close_reason = ?1 \
                 WHERE closed_at IS NULL AND last_activity_at < ?2",
                rusqlite::params![CloseReason::Timer.as_str(), cutoff.to_rfc3339(), now],
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(n)
    }

    // -----------------------------------------------------------------
    // Consolidation support (ANAI-220): close -> summary
    // -----------------------------------------------------------------

    /// Closed episodes that still have no summary and closed no earlier than
    /// `cutoff`, oldest close first, capped at `limit`.
    ///
    /// Returns `(candidates, pending_total)`. `pending_total` is the *unclipped*
    /// count in the same window, so the caller can report what `limit` deferred
    /// rather than silently dropping it — a cap that logs nothing reads as "we
    /// did them all".
    ///
    /// `cutoff` is the caller's, but it is not optional by design: see
    /// [`SUMMARY_LOOKBACK_MINUTES`] for why an unbounded version of this query
    /// is the backfill and not this.
    pub fn awaiting_summary(
        &self,
        cutoff: DateTime<Utc>,
        limit: usize,
    ) -> OpenFangResult<(Vec<Episode>, usize)> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let cutoff = cutoff.to_rfc3339();

        let pending: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episodes \
                 WHERE closed_at IS NOT NULL AND summary IS NULL AND closed_at >= ?1",
                rusqlite::params![cutoff],
                |row| row.get(0),
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let mut stmt = conn
            .prepare(&format!(
                "{SELECT_COLS} WHERE closed_at IS NOT NULL AND summary IS NULL \
                 AND closed_at >= ?1 ORDER BY closed_at ASC LIMIT ?2"
            ))
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let rows = stmt
            .query_map(rusqlite::params![cutoff, limit as i64], row_to_episode)
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let episodes: Vec<Episode> = rows.filter_map(|r| r.ok()).collect();
        Ok((episodes, pending.max(0) as usize))
    }

    /// The episode's material, oldest first: the content of every live memory
    /// row stamped with this episode, capped at `max_rows`.
    ///
    /// Reads `memories` rather than `episodes` — the one place this module
    /// crosses over — because `episode_id` is a promoted, indexed column there
    /// (schema v12) and the episode is the only thing that knows what its own
    /// material is.
    pub fn material(&self, id: Uuid, max_rows: usize) -> OpenFangResult<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT content FROM memories \
                 WHERE episode_id = ?1 AND deleted = 0 ORDER BY created_at ASC LIMIT ?2",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let rows = stmt
            .query_map(rusqlite::params![id.to_string(), max_rows as i64], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Has a summary row already been written for this episode?
    ///
    /// The other half of idempotency: [`Self::set_summary`] guards the
    /// `episodes` column, this guards the recallable `memories` row, so a
    /// re-run (or a future backfill crossing the live path) cannot leave two
    /// summaries of one episode in the corpus. Keyed on `SUMMARY_KIND`, not
    /// `note` — see that constant.
    pub fn has_summary_row(&self, id: Uuid) -> OpenFangResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT 1 FROM memories \
                 WHERE episode_id = ?1 AND kind = ?2 AND deleted = 0 LIMIT 1",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        stmt.exists(rusqlite::params![id.to_string(), SUMMARY_KIND])
            .map_err(|e| OpenFangError::Memory(e.to_string()))
    }

    /// Write a derived title/summary onto an already-closed episode. Returns
    /// whether this call is the one that wrote it.
    ///
    /// `summary IS NULL` in the WHERE clause is the idempotency guard *and* the
    /// precedence rule: a closer's own wrap-up (`close_current`) was written by
    /// someone who was there, and a derived summary must never overwrite it.
    ///
    /// Deliberately separate from the close: the close commits first with a
    /// null summary and this runs afterwards, out of transaction, so a provider
    /// outage costs a summary and never a close.
    pub fn set_summary(
        &self,
        id: Uuid,
        title: Option<&str>,
        summary: &str,
    ) -> OpenFangResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let n = conn
            .execute(
                "UPDATE episodes SET title = COALESCE(title, ?2), summary = ?3 \
                 WHERE id = ?1 AND closed_at IS NOT NULL AND summary IS NULL",
                rusqlite::params![id.to_string(), title, summary],
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(n > 0)
    }

    /// Most recent episodes for an agent, newest first.
    pub fn list_for_agent(&self, agent_id: AgentId, limit: usize) -> OpenFangResult<Vec<Episode>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(&format!(
                "{SELECT_COLS} WHERE agent_id = ?1 ORDER BY opened_at DESC LIMIT ?2"
            ))
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let rows = stmt
            .query_map(
                rusqlite::params![agent_id.0.to_string(), limit as i64],
                row_to_episode,
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    fn is_timed_out(&self, ep: &Episode, now: DateTime<Utc>) -> bool {
        if self.idle_timeout_minutes <= 0 {
            return false;
        }
        now - ep.last_activity_at > Duration::minutes(self.idle_timeout_minutes)
    }
}

const SELECT_COLS: &str = "SELECT id, agent_id, opened_at, last_activity_at, closed_at, title, \
                           summary, close_reason, turn_count FROM episodes";

fn current_open(conn: &Connection, agent_id: AgentId) -> OpenFangResult<Option<Episode>> {
    let mut stmt = conn
        .prepare(&format!(
            "{SELECT_COLS} WHERE agent_id = ?1 AND closed_at IS NULL"
        ))
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
    match stmt.query_row(rusqlite::params![agent_id.0.to_string()], row_to_episode) {
        Ok(ep) => Ok(Some(ep)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(OpenFangError::Memory(e.to_string())),
    }
}

fn insert_open(conn: &Connection, agent_id: AgentId, now: DateTime<Utc>) -> OpenFangResult<Uuid> {
    let id = Uuid::new_v4();
    conn.execute(
        "INSERT INTO episodes (id, agent_id, opened_at, last_activity_at, turn_count) \
         VALUES (?1, ?2, ?3, ?3, 1)",
        rusqlite::params![id.to_string(), agent_id.0.to_string(), now.to_rfc3339()],
    )
    .map_err(|e| OpenFangError::Memory(e.to_string()))?;
    Ok(id)
}

fn close_row(
    conn: &Connection,
    id: Uuid,
    closed_at: DateTime<Utc>,
    reason: CloseReason,
    title: Option<&str>,
    summary: Option<&str>,
) -> OpenFangResult<()> {
    conn.execute(
        "UPDATE episodes SET closed_at = ?2, close_reason = ?3, title = COALESCE(?4, title), \
         summary = COALESCE(?5, summary) WHERE id = ?1 AND closed_at IS NULL",
        rusqlite::params![
            id.to_string(),
            closed_at.to_rfc3339(),
            reason.as_str(),
            title,
            summary
        ],
    )
    .map_err(|e| OpenFangError::Memory(e.to_string()))?;
    Ok(())
}

fn row_to_episode(row: &rusqlite::Row<'_>) -> rusqlite::Result<Episode> {
    let id: String = row.get(0)?;
    let agent_id: String = row.get(1)?;
    let opened_at: String = row.get(2)?;
    let last_activity_at: String = row.get(3)?;
    let closed_at: Option<String> = row.get(4)?;
    let title: Option<String> = row.get(5)?;
    let summary: Option<String> = row.get(6)?;
    let close_reason: Option<String> = row.get(7)?;
    let turn_count: i64 = row.get(8)?;

    Ok(Episode {
        id: Uuid::parse_str(&id).unwrap_or_default(),
        agent_id: AgentId(Uuid::parse_str(&agent_id).unwrap_or_default()),
        opened_at: parse_ts(&opened_at),
        last_activity_at: parse_ts(&last_activity_at),
        closed_at: closed_at.as_deref().map(parse_ts),
        title,
        summary,
        close_reason: close_reason.as_deref().and_then(CloseReason::parse),
        turn_count: turn_count.max(0) as u64,
    })
}

fn parse_ts(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::run_migrations;

    fn store(idle_minutes: i64) -> (EpisodeStore, Arc<Mutex<Connection>>) {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        let shared = Arc::new(Mutex::new(conn));
        (
            EpisodeStore::with_idle_timeout(Arc::clone(&shared), idle_minutes),
            shared,
        )
    }

    /// Force an open episode's activity clock into the past so the timer path
    /// can be exercised without sleeping.
    fn backdate(shared: &Arc<Mutex<Connection>>, id: Uuid, minutes: i64) {
        let ts = (Utc::now() - Duration::minutes(minutes)).to_rfc3339();
        shared
            .lock()
            .unwrap()
            .execute(
                "UPDATE episodes SET opened_at = ?2, last_activity_at = ?2 WHERE id = ?1",
                rusqlite::params![id.to_string(), ts],
            )
            .unwrap();
    }

    #[test]
    fn first_turn_opens_and_subsequent_turns_reuse() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        let e1 = s.ensure_open(a).unwrap();
        let e2 = s.ensure_open(a).unwrap();
        assert_eq!(e1, e2, "a fresh turn must not start a new episode");
        let cur = s.current(a).unwrap().unwrap();
        assert!(cur.is_open());
        assert_eq!(cur.turn_count, 2, "each ensure_open counts one turn");
    }

    #[test]
    fn agents_get_independent_episodes() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        let b = AgentId::new();
        assert_ne!(s.ensure_open(a).unwrap(), s.ensure_open(b).unwrap());
    }

    #[test]
    fn idle_gap_closes_on_timer_and_opens_successor() {
        let (s, c) = store(120);
        let a = AgentId::new();
        let first = s.ensure_open(a).unwrap();
        backdate(&c, first, 121);

        let second = s.ensure_open(a).unwrap();
        assert_ne!(second, first, "a timed-out episode must not be resumed");

        let closed = s.get(first).unwrap().unwrap();
        assert!(!closed.is_open());
        assert_eq!(closed.close_reason, Some(CloseReason::Timer));
        // `closed_at` is when the close RAN, so it is ~now, not the backdated
        // last activity. The work's span is still exact and still recoverable
        // as opened_at..last_activity_at, which the next two asserts pin.
        assert!(
            (Utc::now() - closed.closed_at.unwrap()) < Duration::minutes(1),
            "closed_at must record when the close happened"
        );
        assert!(
            closed.closed_at.unwrap() > closed.last_activity_at,
            "a close cannot precede the activity it closes"
        );
        assert!(
            (Utc::now() - closed.last_activity_at) > Duration::minutes(120),
            "the work's end is still recorded, on last_activity_at"
        );
    }

    #[test]
    fn activity_inside_the_gap_does_not_split() {
        let (s, c) = store(120);
        let a = AgentId::new();
        let first = s.ensure_open(a).unwrap();
        backdate(&c, first, 119);
        assert_eq!(s.ensure_open(a).unwrap(), first);
    }

    #[test]
    fn explicit_close_is_idempotent() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        let id = s.ensure_open(a).unwrap();

        let closed = s
            .close_current(
                a,
                CloseReason::Explicit,
                Some("git trunk cutover"),
                Some("Retired the octopus workflow and moved to trunk."),
            )
            .unwrap();
        assert_eq!(closed, Some(id));

        // Second close is a no-op, not an error.
        assert_eq!(
            s.close_current(a, CloseReason::Explicit, None, None)
                .unwrap(),
            None
        );

        let ep = s.get(id).unwrap().unwrap();
        assert_eq!(ep.close_reason, Some(CloseReason::Explicit));
        assert_eq!(ep.title.as_deref(), Some("git trunk cutover"));
        assert_eq!(
            ep.summary.as_deref(),
            Some("Retired the octopus workflow and moved to trunk."),
            "the closer's wrap-up must survive the round-trip"
        );

        // The next turn starts a fresh episode.
        assert_ne!(s.ensure_open(a).unwrap(), id);
    }

    #[test]
    fn one_open_episode_per_agent_is_a_constraint() {
        let (s, c) = store(120);
        let a = AgentId::new();
        s.ensure_open(a).unwrap();
        // Bypass the store and try to force a second open row. The partial
        // unique index must refuse it — this is the DDL guarantee the whole
        // "DB owns the lifecycle" design rests on.
        let conn = c.lock().unwrap();
        let err = conn.execute(
            "INSERT INTO episodes (id, agent_id, opened_at, last_activity_at, turn_count) \
             VALUES (?1, ?2, ?3, ?3, 1)",
            rusqlite::params![
                Uuid::new_v4().to_string(),
                a.0.to_string(),
                Utc::now().to_rfc3339()
            ],
        );
        assert!(err.is_err(), "second open episode must violate the index");
    }

    #[test]
    fn closed_episodes_do_not_block_the_index() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        for _ in 0..3 {
            s.ensure_open(a).unwrap();
            s.close_current(a, CloseReason::Explicit, None, None)
                .unwrap();
        }
        assert_eq!(s.list_for_agent(a, 10).unwrap().len(), 3);
    }

    #[test]
    fn sweep_closes_only_stale_open_episodes() {
        let (s, c) = store(120);
        let stale_agent = AgentId::new();
        let fresh_agent = AgentId::new();
        let stale = s.ensure_open(stale_agent).unwrap();
        let fresh = s.ensure_open(fresh_agent).unwrap();
        backdate(&c, stale, 300);

        assert_eq!(s.sweep_idle().unwrap(), 1);
        assert_eq!(
            s.get(stale).unwrap().unwrap().close_reason,
            Some(CloseReason::Timer)
        );
        assert!(s.get(fresh).unwrap().unwrap().is_open());
    }

    /// Episodes end when they end. The default store has no timer at all, so a
    /// long silence resumes the same episode rather than fragmenting a thread
    /// that went quiet for a day into a thread per sitting.
    ///
    /// Pinned as a test because it is a product decision living in a constant:
    /// anyone who "restores" the two-hour default has to delete this to do it.
    #[test]
    fn the_idle_timer_is_off_by_default() {
        assert_eq!(DEFAULT_IDLE_TIMEOUT_MINUTES, 0);

        let conn = Arc::new(Mutex::new(Connection::open_in_memory().unwrap()));
        crate::migration::run_migrations(&conn.lock().unwrap()).unwrap();
        let s = EpisodeStore::new(Arc::clone(&conn));
        assert_eq!(s.idle_timeout_minutes(), 0);

        let a = AgentId::new();
        let first = s.ensure_open(a).unwrap();
        backdate(&conn, first, 60 * 24 * 7); // a week of silence
        assert_eq!(
            s.ensure_open(a).unwrap(),
            first,
            "a week of silence must not split the episode"
        );
        assert_eq!(s.sweep_idle().unwrap(), 0, "nothing to reap with no timer");
    }

    #[test]
    fn zero_timeout_disables_timer_close() {
        let (s, c) = store(0);
        let a = AgentId::new();
        let id = s.ensure_open(a).unwrap();
        backdate(&c, id, 10_000);
        assert_eq!(s.ensure_open(a).unwrap(), id, "timer close must be off");
        assert_eq!(s.sweep_idle().unwrap(), 0);
    }

    #[test]
    fn close_reason_round_trips() {
        for r in [
            CloseReason::TopicSwitch,
            CloseReason::Explicit,
            CloseReason::Timer,
            CloseReason::Abandoned,
        ] {
            assert_eq!(CloseReason::parse(r.as_str()), Some(r));
        }
        assert_eq!(CloseReason::parse("nonsense"), None);
    }

    #[test]
    fn status_reports_nothing_open_for_a_fresh_agent() {
        let (s, _c) = store(120);
        let st = s.status(AgentId::new(), 3).unwrap();
        assert!(st.current.is_none());
        assert_eq!(st.idle_minutes, None);
        assert_eq!(st.minutes_until_timer_close, None);
        assert!(st.recent.is_empty());
        assert_eq!(st.idle_timeout_minutes, 120);
    }

    #[test]
    fn status_counts_down_the_idle_gap() {
        let (s, c) = store(120);
        let a = AgentId::new();
        let id = s.ensure_open(a).unwrap();
        backdate(&c, id, 90);

        let st = s.status(a, 0).unwrap();
        assert_eq!(st.current.unwrap().id, id);
        assert_eq!(st.idle_minutes, Some(90));
        assert_eq!(st.minutes_until_timer_close, Some(30));
    }

    /// An episode past its gap is closed by the NEXT turn, not retroactively,
    /// so the countdown floors at 0. A negative number would read as a
    /// diagnostic ("7 minutes overdue") the agent cannot act on, and would
    /// invite callers to branch on the sign.
    #[test]
    fn status_countdown_floors_at_zero_past_the_gap() {
        let (s, c) = store(120);
        let a = AgentId::new();
        let id = s.ensure_open(a).unwrap();
        backdate(&c, id, 500);
        assert_eq!(s.status(a, 0).unwrap().minutes_until_timer_close, Some(0));
    }

    #[test]
    fn status_omits_the_countdown_when_the_timer_is_disabled() {
        let (s, _c) = store(0);
        let a = AgentId::new();
        s.ensure_open(a).unwrap();
        let st = s.status(a, 0).unwrap();
        assert!(st.current.is_some());
        assert_eq!(st.minutes_until_timer_close, None, "timer is off");
    }

    /// The history tail must not include the episode the caller already has in
    /// `current` — that is how a "recent" list quietly becomes off-by-one.
    #[test]
    fn status_history_excludes_the_open_episode() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        for i in 0..3 {
            s.ensure_open(a).unwrap();
            s.close_current(a, CloseReason::Explicit, Some(&format!("ep{i}")), None)
                .unwrap();
        }
        let open = s.ensure_open(a).unwrap();

        let st = s.status(a, 2).unwrap();
        assert_eq!(st.current.unwrap().id, open);
        assert_eq!(st.recent.len(), 2);
        assert!(
            st.recent.iter().all(|ep| !ep.is_open()),
            "the open episode must not appear in its own history"
        );
    }

    /// A timer close leaves `summary` NULL. Nothing is present to write one,
    /// and a fabricated wrap-up would be an ungrounded claim in a tier that
    /// exists to hold grounded ones.
    #[test]
    fn timer_close_writes_no_summary() {
        let (s, c) = store(120);
        let a = AgentId::new();
        let first = s.ensure_open(a).unwrap();
        backdate(&c, first, 121);
        s.ensure_open(a).unwrap();

        let closed = s.get(first).unwrap().unwrap();
        assert_eq!(closed.close_reason, Some(CloseReason::Timer));
        assert_eq!(closed.summary, None);
        assert_eq!(closed.title, None);
    }

    // -----------------------------------------------------------------
    // ANAI-220: close -> summary support
    // -----------------------------------------------------------------

    /// Stamp a live memory row onto an episode. Mirrors the column list
    /// `SemanticStore::remember_sqlite` writes, so the reads under test see the
    /// same shape the capture path produces.
    fn insert_memory(
        shared: &Arc<Mutex<Connection>>,
        agent: AgentId,
        episode_id: Uuid,
        kind: Option<&str>,
        content: &str,
        created_at: DateTime<Utc>,
    ) {
        shared
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO memories (id, agent_id, content, source, scope, confidence, \
                 metadata, created_at, accessed_at, access_count, deleted, embedding, \
                 episode_id, kind) \
                 VALUES (?1, ?2, ?3, '\"observation\"', 'episodic', 1.0, '{}', ?4, ?4, 0, 0, \
                 NULL, ?5, ?6)",
                rusqlite::params![
                    Uuid::new_v4().to_string(),
                    agent.0.to_string(),
                    content,
                    created_at.to_rfc3339(),
                    episode_id.to_string(),
                    kind,
                ],
            )
            .unwrap();
    }

    /// Backdate a CLOSED episode's close so the lookback window can be
    /// exercised without waiting an hour.
    fn backdate_close(shared: &Arc<Mutex<Connection>>, id: Uuid, minutes: i64) {
        let ts = (Utc::now() - Duration::minutes(minutes)).to_rfc3339();
        shared
            .lock()
            .unwrap()
            .execute(
                "UPDATE episodes SET closed_at = ?2 WHERE id = ?1",
                rusqlite::params![id.to_string(), ts],
            )
            .unwrap();
    }

    fn close_now(s: &EpisodeStore, agent: AgentId) -> Uuid {
        s.ensure_open(agent).unwrap();
        s.close_current(agent, CloseReason::Explicit, None, None)
            .unwrap()
            .unwrap()
    }

    /// The candidate set is closed-and-unsummarised only. An open episode is
    /// still being written; a summarised one is done.
    #[test]
    fn awaiting_summary_takes_only_closed_unsummarised_episodes() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        let b = AgentId::new();
        let d = AgentId::new();

        let bare = close_now(&s, a);
        s.ensure_open(b).unwrap(); // open: not a candidate
        let authored = s.ensure_open(d).unwrap();
        s.close_current(d, CloseReason::Explicit, Some("t"), Some("the closer's own"))
            .unwrap();

        let (found, pending) = s.awaiting_summary(Utc::now() - Duration::hours(1), 10).unwrap();
        assert_eq!(pending, 1);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, bare);
        assert_ne!(found[0].id, authored);
    }

    /// **The backfill guard.** Without the cutoff this selector is
    /// `closed_at IS NOT NULL AND summary IS NULL`, which is every null-summary
    /// episode ever closed. Deleting this test is how B silently becomes a
    /// fleet-wide backfill on its first armed tick.
    #[test]
    fn awaiting_summary_is_bounded_by_close_recency_not_all_history() {
        let (s, c) = store(120);
        let a = AgentId::new();
        let ancient = close_now(&s, a);
        backdate_close(&c, ancient, SUMMARY_LOOKBACK_MINUTES + 5);

        let cutoff = Utc::now() - Duration::minutes(SUMMARY_LOOKBACK_MINUTES);
        let (found, pending) = s.awaiting_summary(cutoff, 10).unwrap();
        assert!(found.is_empty(), "a stale close is backfill work, not live work");
        assert_eq!(pending, 0);
    }

    /// A cap must report what it deferred. `pending` is the unclipped count so
    /// the caller can say "8 summarized, 4 deferred" rather than implying it
    /// did them all.
    #[test]
    fn awaiting_summary_reports_what_the_cap_deferred() {
        let (s, _c) = store(120);
        for _ in 0..5 {
            close_now(&s, AgentId::new());
        }
        let (found, pending) = s.awaiting_summary(Utc::now() - Duration::hours(1), 3).unwrap();
        assert_eq!(found.len(), 3, "the cap clips the batch");
        assert_eq!(pending, 5, "but not the count the caller logs");
    }

    /// Material is the episode's linked rows, oldest first and capped — never
    /// `turn_count`, which counts turns and not stored content.
    #[test]
    fn material_is_linked_rows_oldest_first_and_capped() {
        let (s, c) = store(120);
        let a = AgentId::new();
        let ep = s.ensure_open(a).unwrap();
        let other = s.ensure_open(AgentId::new()).unwrap();

        let t0 = Utc::now() - Duration::minutes(10);
        insert_memory(&c, a, ep, None, "first", t0);
        insert_memory(&c, a, ep, Some("note"), "second", t0 + Duration::minutes(1));
        insert_memory(&c, a, ep, None, "third", t0 + Duration::minutes(2));
        insert_memory(&c, a, other, None, "not mine", t0);

        assert_eq!(s.material(ep, 40).unwrap(), vec!["first", "second", "third"]);
        assert_eq!(s.material(ep, 2).unwrap(), vec!["first", "second"]);
    }

    /// An episode whose agent talked for a dozen turns but stored nothing has
    /// no material. The floor is on rows, and this is the population it exists
    /// for: measured on the live fleet, 14 turns / 0 linked rows.
    #[test]
    fn a_high_turn_count_episode_can_still_have_no_material() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        let ep = s.ensure_open(a).unwrap();
        for _ in 0..13 {
            s.ensure_open(a).unwrap();
        }
        assert_eq!(s.current(a).unwrap().unwrap().turn_count, 14);
        assert!(s.material(ep, 40).unwrap().len() < MIN_MATERIAL_ROWS);
    }

    /// The idempotency key must not collide with `memory_note`. If it did,
    /// every episode the agent jotted a note in would be read as
    /// already-summarised — skipping exactly the episodes richest in material.
    #[test]
    fn a_hand_written_note_is_not_a_summary_row() {
        let (s, c) = store(120);
        let a = AgentId::new();
        let ep = s.ensure_open(a).unwrap();
        insert_memory(&c, a, ep, Some("note"), "an agent's own note", Utc::now());
        assert!(!s.has_summary_row(ep).unwrap());

        insert_memory(&c, a, ep, Some(SUMMARY_KIND), "the derived summary", Utc::now());
        assert!(s.has_summary_row(ep).unwrap());
    }

    /// Re-running over an already-summarised episode is a no-op. This is the
    /// property that makes a later backfill safe to point at the live path.
    #[test]
    fn set_summary_writes_once_and_then_no_ops() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        let ep = close_now(&s, a);

        assert!(s.set_summary(ep, Some("landed B"), "wired close to summary").unwrap());
        assert!(
            !s.set_summary(ep, Some("second try"), "a different summary").unwrap(),
            "a re-run must not rewrite a summary"
        );

        let row = s.get(ep).unwrap().unwrap();
        assert_eq!(row.title.as_deref(), Some("landed B"));
        assert_eq!(row.summary.as_deref(), Some("wired close to summary"));
    }

    /// A closer who was there beats a model that was not. `close_current`'s
    /// wrap-up is authored; a derived summary must never overwrite it.
    #[test]
    fn set_summary_never_overwrites_the_closers_own_wrap_up() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        s.ensure_open(a).unwrap();
        let ep = s
            .close_current(a, CloseReason::Explicit, Some("mine"), Some("I was there"))
            .unwrap()
            .unwrap();

        assert!(!s.set_summary(ep, Some("derived"), "I was not").unwrap());
        let row = s.get(ep).unwrap().unwrap();
        assert_eq!(row.summary.as_deref(), Some("I was there"));
        assert_eq!(row.title.as_deref(), Some("mine"));
    }

    /// An open episode is not summarisable — the close is what makes the
    /// interval final.
    #[test]
    fn set_summary_refuses_an_open_episode() {
        let (s, _c) = store(120);
        let a = AgentId::new();
        let ep = s.ensure_open(a).unwrap();
        assert!(!s.set_summary(ep, None, "premature").unwrap());
        assert_eq!(s.get(ep).unwrap().unwrap().summary, None);
    }
}
