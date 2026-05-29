//! Discord Gateway adapter for the OpenFang channel bridge.
//!
//! Uses Discord Gateway WebSocket (v10) for receiving messages and the REST API
//! for sending responses. No external Discord crate — just `tokio-tungstenite` + `reqwest`.

use crate::types::{
    split_message, ChannelAdapter, ChannelContent, ChannelMessage, ChannelType, ChannelUser,
    ResolutionError,
};
use async_trait::async_trait;
use futures::{SinkExt, Stream, StreamExt};
use std::collections::{HashMap, HashSet, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch, Mutex, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use zeroize::Zeroizing;

const DISCORD_API_BASE: &str = "https://discord.com/api/v10";
const MAX_BACKOFF: Duration = Duration::from_secs(60);
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const DISCORD_MSG_LIMIT: usize = 2000;
/// Maximum number of seen message IDs kept in the dedup set.
/// MESSAGE_UPDATE (embed resolution) events arrive within seconds of the
/// original CREATE; entries older than this cap are safe to discard.
const MAX_DEDUP_MSG_IDS: usize = 2_000;

// --- ANAI-55 recipient resolution ------------------------------------------
/// Dedicated `tracing` target for the recipient-resolution audit log.
///
/// Kept separate from the message-payload log target so a log scrape cannot
/// trivially correlate "agent X resolved recipient Y" with the message body
/// (security review observability ask).
const RESOLVER_AUDIT_TARGET: &str = "openfang_channels::discord::resolver_audit";

/// Window for the resolution-failure burst counter.
const RESOLVER_FAIL_WINDOW: Duration = Duration::from_secs(60);

/// Failures within `RESOLVER_FAIL_WINDOW` above this count emit a single
/// `warn!` (rate-limited via the deque so we don't spam at high rates).
const RESOLVER_FAIL_WARN_THRESHOLD: usize = 20;

/// Discord Gateway opcodes.
mod opcode {
    pub const DISPATCH: u64 = 0;
    pub const HEARTBEAT: u64 = 1;
    pub const IDENTIFY: u64 = 2;
    pub const RESUME: u64 = 6;
    pub const RECONNECT: u64 = 7;
    pub const INVALID_SESSION: u64 = 9;
    pub const HELLO: u64 = 10;
    pub const HEARTBEAT_ACK: u64 = 11;
}

/// Build a Discord gateway heartbeat (opcode 1) payload.
///
/// Per the Discord gateway spec, the payload `d` field is the last received
/// dispatch sequence number, or `null` if no dispatch has been received yet.
/// See: <https://discord.com/developers/docs/topics/gateway#sending-heartbeats>
fn build_heartbeat_payload(last_sequence: Option<u64>) -> serde_json::Value {
    serde_json::json!({
        "op": opcode::HEARTBEAT,
        "d": last_sequence,
    })
}

/// Discord Gateway adapter using WebSocket.
pub struct DiscordAdapter {
    /// SECURITY: Bot token is zeroized on drop to prevent memory disclosure.
    token: Zeroizing<String>,
    client: reqwest::Client,
    allowed_guilds: Vec<String>,
    allowed_users: Vec<String>,
    ignore_bots: bool,
    intents: u64,
    /// Auto-thread behavior: "true", "false", or "smart"
    auto_thread: String,
    shutdown_tx: Arc<watch::Sender<bool>>,
    shutdown_rx: watch::Receiver<bool>,
    /// Bot's own user ID (populated after READY event).
    bot_user_id: Arc<RwLock<Option<String>>>,
    /// Session ID for resume (populated after READY event).
    session_id: Arc<RwLock<Option<String>>>,
    /// Resume gateway URL.
    resume_gateway_url: Arc<RwLock<Option<String>>>,
    /// Thread channel IDs created by this bot (thread_id → parent_channel_id).
    /// Used to detect when incoming messages are inside a bot-created thread.
    created_thread_ids: Arc<RwLock<HashMap<String, String>>>,
    /// Message IDs seen via MESSAGE_CREATE (used to drop duplicate MESSAGE_UPDATE events).
    /// Populated immediately when MESSAGE_CREATE is forwarded — before bridge processing —
    /// to eliminate the race window where MESSAGE_UPDATE arrives before thread creation completes.
    threaded_message_ids: Arc<RwLock<HashSet<String>>>,

    // --- ANAI-55 recipient resolution caches -------------------------------
    /// `username` → `user_id`. Populated passively on every inbound
    /// MESSAGE_CREATE. Services `@username` DM resolution only — bare names
    /// are refused (see [`ResolutionError::BareNameDmRefused`]).
    /// Last-write-wins on collision; overwrites emit a `debug!` line.
    username_to_user_id: Arc<RwLock<HashMap<String, String>>>,

    /// `(guild_id, channel_name)` → `channel_id`. Keyed by guild to prevent
    /// cross-guild `#name` collisions (security F2). Populated from
    /// GUILD_CREATE, GUILD_UPDATE, CHANNEL_CREATE, CHANNEL_UPDATE; evicted
    /// on CHANNEL_DELETE.
    channel_name_to_channel_id: Arc<RwLock<HashMap<(String, String), String>>>,

    /// `user_id` → DM channel id. Populated on first successful DM-open via
    /// `POST /users/@me/channels`. 4xx fails closed; no auto-retry.
    user_id_to_dm_channel_id: Arc<RwLock<HashMap<String, String>>>,

    /// Sliding-window timestamps of recent resolution failures (used for the
    /// burst-warn defense-in-depth signal).
    resolution_failures: Arc<RwLock<VecDeque<Instant>>>,
}

impl DiscordAdapter {
    pub fn new(
        token: String,
        allowed_guilds: Vec<String>,
        allowed_users: Vec<String>,
        ignore_bots: bool,
        intents: u64,
        auto_thread: String,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            token: Zeroizing::new(token),
            client: reqwest::Client::new(),
            allowed_guilds,
            allowed_users,
            ignore_bots,
            intents,
            auto_thread,
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
            bot_user_id: Arc::new(RwLock::new(None)),
            session_id: Arc::new(RwLock::new(None)),
            resume_gateway_url: Arc::new(RwLock::new(None)),
            created_thread_ids: Arc::new(RwLock::new(HashMap::new())),
            threaded_message_ids: Arc::new(RwLock::new(HashSet::new())),
            username_to_user_id: Arc::new(RwLock::new(HashMap::new())),
            channel_name_to_channel_id: Arc::new(RwLock::new(HashMap::new())),
            user_id_to_dm_channel_id: Arc::new(RwLock::new(HashMap::new())),
            resolution_failures: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Get the WebSocket gateway URL from the Discord API.
    async fn get_gateway_url(&self) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{DISCORD_API_BASE}/gateway/bot");
        let resp: serde_json::Value = self
            .client
            .get(&url)
            .header("Authorization", format!("Bot {}", self.token.as_str()))
            .send()
            .await?
            .json()
            .await?;

        let ws_url = resp["url"]
            .as_str()
            .ok_or("Missing 'url' in gateway response")?;

        Ok(format!("{ws_url}/?v=10&encoding=json"))
    }

    /// Send a message to a Discord channel via REST API.
    async fn api_send_message(
        &self,
        channel_id: &str,
        text: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{DISCORD_API_BASE}/channels/{channel_id}/messages");
        let chunks = split_message(text, DISCORD_MSG_LIMIT);

        for chunk in chunks {
            let body = serde_json::json!({ "content": chunk });
            let resp = self
                .client
                .post(&url)
                .header("Authorization", format!("Bot {}", self.token.as_str()))
                .json(&body)
                .send()
                .await?;

            if !resp.status().is_success() {
                let body_text = resp.text().await.unwrap_or_default();
                warn!("Discord sendMessage failed: {body_text}");
            }
        }
        Ok(())
    }

    /// Send typing indicator to a Discord channel.
    async fn api_send_typing(&self, channel_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{DISCORD_API_BASE}/channels/{channel_id}/typing");
        let _ = self
            .client
            .post(&url)
            .header("Authorization", format!("Bot {}", self.token.as_str()))
            .send()
            .await?;
        Ok(())
    }

    /// Open (or fetch) the DM channel for a given user snowflake.
    ///
    /// On success returns the DM channel id and inserts it into
    /// `user_id_to_dm_channel_id` for future calls. On HTTP 4xx the status
    /// code is returned to the caller for fail-closed handling — no
    /// auto-retry (see ANAI-55 §3.2 step 6). Transport failures map to `0`.
    async fn api_open_dm_channel(&self, user_id: &str) -> Result<String, u16> {
        let url = format!("{DISCORD_API_BASE}/users/@me/channels");
        let body = serde_json::json!({ "recipient_id": user_id });
        let resp = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bot {}", self.token.as_str()))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!("discord DM-open transport error for {user_id}: {e}");
                return Err(0);
            }
        };
        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            return Err(status);
        }
        let v: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                warn!("discord DM-open returned non-JSON for {user_id}: {e}");
                return Err(status);
            }
        };
        let dm_id = match v["id"].as_str() {
            Some(s) => s.to_string(),
            None => {
                warn!("discord DM-open missing `id` field for {user_id}");
                return Err(status);
            }
        };
        self.user_id_to_dm_channel_id
            .write()
            .await
            .insert(user_id.to_string(), dm_id.clone());
        Ok(dm_id)
    }

    /// ANAI-55 resolver body. Returns `(platform_id, via_label)` on success.
    async fn resolve_recipient_inner(
        &self,
        recipient: &str,
    ) -> Result<(String, &'static str), ResolutionError> {
        let r = recipient.trim();
        if r.is_empty() {
            return Err(ResolutionError::UnknownRecipient {
                recipient: recipient.to_string(),
            });
        }

        // 1. Raw snowflake.
        if is_snowflake(r) {
            return Ok((r.to_string(), "snowflake"));
        }

        // 2. Channel mention <#id>.
        if let Some(id) = strip_channel_mention(r) {
            if is_snowflake(&id) {
                return Ok((id, "channel_mention"));
            }
        }

        // 3. User mention <@id> / <@!id> — DM-open with the id.
        if let Some(id) = strip_user_mention(r) {
            if is_snowflake(&id) {
                return self
                    .resolve_user_id_to_dm(&id)
                    .await
                    .map(|d| (d, "user_mention+dm_open"));
            }
        }

        // 4. `#name` or bare channel-shaped name.
        let (had_hash, name_part) = if let Some(stripped) = r.strip_prefix('#') {
            (true, stripped)
        } else {
            (false, r)
        };

        if !name_part.is_empty() && is_channel_name_shape(name_part) {
            let cache = self.channel_name_to_channel_id.read().await;
            let mut hits: Vec<(String, String)> = cache
                .iter()
                .filter(|((_gid, n), _cid)| n == name_part)
                .map(|((gid, _n), cid)| (gid.clone(), cid.clone()))
                .collect();
            drop(cache);
            match hits.len() {
                1 => {
                    let (_gid, cid) = hits.pop().unwrap();
                    return Ok((cid, "channel_name"));
                }
                n if n > 1 => {
                    let guilds: Vec<String> = hits.into_iter().map(|(g, _)| g).collect();
                    return Err(ResolutionError::AmbiguousChannel {
                        name: name_part.to_string(),
                        guilds,
                    });
                }
                _ => {
                    if had_hash {
                        return Err(ResolutionError::UnknownRecipient {
                            recipient: recipient.to_string(),
                        });
                    }
                }
            }
        }

        // 5. `@username` — DM path.
        if let Some(uname) = r.strip_prefix('@') {
            if !uname.is_empty() && is_username_shape(uname) {
                let user_id = self.username_to_user_id.read().await.get(uname).cloned();
                return match user_id {
                    Some(uid) => self
                        .resolve_user_id_to_dm(&uid)
                        .await
                        .map(|d| (d, "username_cache+dm_open")),
                    None => Err(ResolutionError::UnknownRecipient {
                        recipient: recipient.to_string(),
                    }),
                };
            }
        }

        // 6. Bare name — refused as a DM target (ANAI-55 F1).
        Err(ResolutionError::BareNameDmRefused {
            name: r.to_string(),
        })
    }

    /// Step 6 of the resolver: cache-or-open the DM channel for a user id.
    async fn resolve_user_id_to_dm(&self, user_id: &str) -> Result<String, ResolutionError> {
        if let Some(dm) = self
            .user_id_to_dm_channel_id
            .read()
            .await
            .get(user_id)
            .cloned()
        {
            return Ok(dm);
        }
        self.api_open_dm_channel(user_id)
            .await
            .map_err(|status| ResolutionError::DmOpenFailed {
                user_id: user_id.to_string(),
                status,
            })
    }

    /// Record a resolution failure for the sliding-window burst-warn signal.
    /// Returns `true` the single time the threshold is crossed within a window.
    async fn note_resolution_failure(&self) -> bool {
        let now = Instant::now();
        let mut q = self.resolution_failures.write().await;
        while let Some(front) = q.front() {
            if now.duration_since(*front) > RESOLVER_FAIL_WINDOW {
                q.pop_front();
            } else {
                break;
            }
        }
        q.push_back(now);
        q.len() == RESOLVER_FAIL_WARN_THRESHOLD + 1
    }

    /// Create a thread from a message in a Discord channel.
    async fn api_create_thread(
        &self,
        channel_id: &str,
        message_id: &str,
        name: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!(
            "{DISCORD_API_BASE}/channels/{channel_id}/messages/{message_id}/threads",
            channel_id = channel_id,
            message_id = message_id
        );
        let body = serde_json::json!({
            "name": name,
            "auto_archive_duration": 1440 // 24 hours
        });
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bot {}", self.token.as_str()))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(format!("Discord createThread failed: {}", body_text).into());
        }

        let response: serde_json::Value = resp.json().await?;
        let thread_id = response["id"].as_str().unwrap_or("").to_string();

        // Track thread_id → parent channel_id so we can recognise messages
        // that arrive inside this thread.
        if !thread_id.is_empty() {
            self.created_thread_ids
                .write()
                .await
                .insert(thread_id.clone(), channel_id.to_string());
        }

        Ok(thread_id)
    }

    /// Send a message to an existing thread.
    /// Discord threads are channels — post directly to channels/{thread_id}/messages.
    async fn api_send_thread_message(
        &self,
        _channel_id: &str,
        thread_id: &str,
        text: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{DISCORD_API_BASE}/channels/{thread_id}/messages");
        let chunks = split_message(text, DISCORD_MSG_LIMIT);

        for chunk in chunks {
            let body = serde_json::json!({ "content": chunk });
            let resp = self
                .client
                .post(&url)
                .header("Authorization", format!("Bot {}", self.token.as_str()))
                .json(&body)
                .send()
                .await?;

            if !resp.status().is_success() {
                let body_text = resp.text().await.unwrap_or_default();
                warn!("Discord sendThreadMessage failed: {body_text}");
            }
        }
        Ok(())
    }
}

#[async_trait]
impl ChannelAdapter for DiscordAdapter {
    fn name(&self) -> &str {
        "discord"
    }

    fn channel_type(&self) -> ChannelType {
        ChannelType::Discord
    }

    async fn should_auto_thread(&self, message: &ChannelMessage) -> Option<String> {
        // Only auto-thread in group channels (servers), not DMs
        if !message.is_group {
            return None;
        }

        // Check auto_thread mode
        match self.auto_thread.as_str() {
            "true" => Some(thread_name_from_message(message)),
            "false" => None,
            "smart" => {
                // Only create thread if bot was @mentioned
                let was_mentioned = message
                    .metadata
                    .get("was_mentioned")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if was_mentioned {
                    Some(thread_name_from_message(message))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    async fn start(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = ChannelMessage> + Send>>, Box<dyn std::error::Error>>
    {
        let gateway_url = self.get_gateway_url().await?;
        info!("Discord gateway URL obtained");

        let (tx, rx) = mpsc::channel::<ChannelMessage>(256);

        let token = self.token.clone();
        let intents = self.intents;
        let allowed_guilds = self.allowed_guilds.clone();
        let allowed_users = self.allowed_users.clone();
        let ignore_bots = self.ignore_bots;
        let bot_user_id = self.bot_user_id.clone();
        let session_id_store = self.session_id.clone();
        let resume_url_store = self.resume_gateway_url.clone();
        let created_thread_ids = self.created_thread_ids.clone();
        let threaded_message_ids = self.threaded_message_ids.clone();
        let username_to_user_id = self.username_to_user_id.clone();
        let channel_name_to_channel_id = self.channel_name_to_channel_id.clone();
        let mut shutdown = self.shutdown_rx.clone();

        tokio::spawn(async move {
            let mut backoff = INITIAL_BACKOFF;
            let mut connect_url = gateway_url;
            // Sequence persists across reconnections for RESUME
            let sequence: Arc<RwLock<Option<u64>>> = Arc::new(RwLock::new(None));

            loop {
                if *shutdown.borrow() {
                    break;
                }

                info!("Connecting to Discord gateway...");

                let ws_result = tokio_tungstenite::connect_async(&connect_url).await;
                let ws_stream = match ws_result {
                    Ok((stream, _)) => stream,
                    Err(e) => {
                        warn!("Discord gateway connection failed: {e}, retrying in {backoff:?}");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(MAX_BACKOFF);
                        continue;
                    }
                };

                backoff = INITIAL_BACKOFF;
                info!("Discord gateway connected");

                let (ws_tx_raw, mut ws_rx) = ws_stream.split();
                // Wrap the sink so the periodic heartbeat task and the inner
                // loop can both write to it.
                let ws_tx = Arc::new(Mutex::new(ws_tx_raw));
                let mut heartbeat_handle: Option<JoinHandle<()>> = None;
                // Tracks whether the most recent heartbeat we sent has been
                // ACKed (opcode 11). Initialized to `true` so the first
                // heartbeat is always allowed to fire.
                let heartbeat_acked = Arc::new(AtomicBool::new(true));

                // Inner message loop — returns true if we should reconnect
                let should_reconnect = 'inner: loop {
                    let msg = tokio::select! {
                        msg = ws_rx.next() => msg,
                        _ = shutdown.changed() => {
                            if *shutdown.borrow() {
                                info!("Discord shutdown requested");
                                if let Some(h) = heartbeat_handle.take() {
                                    h.abort();
                                }
                                let _ = ws_tx.lock().await.close().await;
                                return;
                            }
                            continue;
                        }
                    };

                    let msg = match msg {
                        Some(Ok(m)) => m,
                        Some(Err(e)) => {
                            warn!("Discord WebSocket error: {e}");
                            break 'inner true;
                        }
                        None => {
                            info!("Discord WebSocket closed");
                            break 'inner true;
                        }
                    };

                    let text = match msg {
                        tokio_tungstenite::tungstenite::Message::Text(t) => t,
                        tokio_tungstenite::tungstenite::Message::Close(_) => {
                            info!("Discord gateway closed by server");
                            break 'inner true;
                        }
                        _ => continue,
                    };

                    let payload: serde_json::Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Discord: failed to parse gateway message: {e}");
                            continue;
                        }
                    };

                    let op = payload["op"].as_u64().unwrap_or(999);

                    // Update sequence number from any payload that carries one
                    // (typically dispatch events, opcode 0).
                    if let Some(s) = payload["s"].as_u64() {
                        *sequence.write().await = Some(s);
                    }

                    match op {
                        opcode::HELLO => {
                            let interval =
                                payload["d"]["heartbeat_interval"].as_u64().unwrap_or(45000);
                            debug!("Discord HELLO: heartbeat_interval={interval}ms");

                            // Spawn the periodic heartbeat task BEFORE we send
                            // IDENTIFY/RESUME, per the Discord gateway flow.
                            // Abort any stale handle from a previous attempt
                            // first (defensive — should normally be None here).
                            if let Some(h) = heartbeat_handle.take() {
                                h.abort();
                            }
                            heartbeat_acked.store(true, Ordering::Relaxed);
                            let hb_sink = ws_tx.clone();
                            let hb_seq = sequence.clone();
                            let hb_acked = heartbeat_acked.clone();
                            let mut hb_shutdown = shutdown.clone();
                            heartbeat_handle = Some(tokio::spawn(async move {
                                let mut ticker =
                                    tokio::time::interval(Duration::from_millis(interval));
                                // Skip the immediate first tick — we want to
                                // wait one full interval before the first beat.
                                ticker.tick().await;
                                loop {
                                    tokio::select! {
                                        _ = ticker.tick() => {}
                                        _ = hb_shutdown.changed() => {
                                            if *hb_shutdown.borrow() {
                                                return;
                                            }
                                            continue;
                                        }
                                    }

                                    // If the previous heartbeat was never
                                    // ACKed, the connection is zombied — close
                                    // the sink so the read loop sees EOF and
                                    // triggers a reconnect (Discord spec).
                                    if !hb_acked.swap(false, Ordering::Relaxed) {
                                        warn!(
                                            "Discord: previous heartbeat not ACKed, \
                                             forcing reconnect"
                                        );
                                        let _ = hb_sink.lock().await.close().await;
                                        return;
                                    }

                                    let seq = *hb_seq.read().await;
                                    let payload = build_heartbeat_payload(seq);
                                    let text = match serde_json::to_string(&payload) {
                                        Ok(s) => s,
                                        Err(e) => {
                                            error!("Discord: failed to serialize heartbeat: {e}");
                                            return;
                                        }
                                    };
                                    let send_res = hb_sink
                                        .lock()
                                        .await
                                        .send(tokio_tungstenite::tungstenite::Message::Text(text))
                                        .await;
                                    if let Err(e) = send_res {
                                        warn!("Discord: failed to send heartbeat: {e}");
                                        return;
                                    }
                                    debug!("Discord heartbeat sent (seq={:?})", seq);
                                }
                            }));

                            // Try RESUME if we have a session, otherwise IDENTIFY
                            let has_session = session_id_store.read().await.is_some();
                            let has_seq = sequence.read().await.is_some();

                            let gateway_msg = if has_session && has_seq {
                                let sid = session_id_store.read().await.clone().unwrap();
                                let seq = *sequence.read().await;
                                info!("Discord: sending RESUME (session={sid})");
                                serde_json::json!({
                                    "op": opcode::RESUME,
                                    "d": {
                                        "token": token.as_str(),
                                        "session_id": sid,
                                        "seq": seq
                                    }
                                })
                            } else {
                                info!("Discord: sending IDENTIFY");
                                serde_json::json!({
                                    "op": opcode::IDENTIFY,
                                    "d": {
                                        "token": token.as_str(),
                                        "intents": intents,
                                        "properties": {
                                            "os": "linux",
                                            "browser": "openfang",
                                            "device": "openfang"
                                        }
                                    }
                                })
                            };

                            if let Err(e) = ws_tx
                                .lock()
                                .await
                                .send(tokio_tungstenite::tungstenite::Message::Text(
                                    serde_json::to_string(&gateway_msg).unwrap(),
                                ))
                                .await
                            {
                                error!("Discord: failed to send IDENTIFY/RESUME: {e}");
                                break 'inner true;
                            }
                        }

                        opcode::DISPATCH => {
                            let event_name = payload["t"].as_str().unwrap_or("");
                            let d = &payload["d"];

                            match event_name {
                                "READY" => {
                                    let user_id =
                                        d["user"]["id"].as_str().unwrap_or("").to_string();
                                    let username =
                                        d["user"]["username"].as_str().unwrap_or("unknown");
                                    let sid = d["session_id"].as_str().unwrap_or("").to_string();
                                    let resume_url =
                                        d["resume_gateway_url"].as_str().unwrap_or("").to_string();

                                    *bot_user_id.write().await = Some(user_id.clone());
                                    *session_id_store.write().await = Some(sid);
                                    if !resume_url.is_empty() {
                                        *resume_url_store.write().await = Some(resume_url);
                                    }

                                    info!("Discord bot ready: {username} ({user_id})");
                                }

                                "MESSAGE_CREATE" | "MESSAGE_UPDATE" => {
                                    // Passively populate the username cache
                                    // for ANAI-55 `@username` resolution.
                                    // Done before parse so we still learn the
                                    // mapping even when the message itself is
                                    // dropped by allow-list filters downstream.
                                    if event_name == "MESSAGE_CREATE" {
                                        if let (Some(uname), Some(uid)) = (
                                            d["author"]["username"].as_str(),
                                            d["author"]["id"].as_str(),
                                        ) {
                                            let mut cache = username_to_user_id.write().await;
                                            if let Some(prev) = cache.get(uname) {
                                                if prev != uid {
                                                    debug!(
                                                        "discord username cache overwrite:                                                          {uname} {prev} -> {uid}"
                                                    );
                                                }
                                            }
                                            cache.insert(uname.to_string(), uid.to_string());
                                        }
                                    }

                                    if let Some(msg) = parse_discord_message(
                                        d,
                                        &bot_user_id,
                                        &allowed_guilds,
                                        &allowed_users,
                                        ignore_bots,
                                        &created_thread_ids,
                                    )
                                    .await
                                    {
                                        // MESSAGE_UPDATE must be suppressed if we already
                                        // forwarded a MESSAGE_CREATE for this message ID.
                                        // The check uses `seen_message_ids` (tracked below)
                                        // which is populated the moment MESSAGE_CREATE is
                                        // forwarded — before the bridge even processes it.
                                        // This closes the race window where MESSAGE_UPDATE
                                        // arrives before adapter.create_thread() completes.
                                        if event_name == "MESSAGE_UPDATE"
                                            && threaded_message_ids
                                                .read()
                                                .await
                                                .contains(&msg.platform_message_id)
                                        {
                                            debug!(
                                                "Discord MESSAGE_UPDATE skipped (already seen {})",
                                                msg.platform_message_id
                                            );
                                            continue;
                                        }

                                        debug!(
                                            "Discord {event_name} from {}: {:?}",
                                            msg.sender.display_name, msg.content
                                        );

                                        // Mark this message as seen immediately so any
                                        // concurrent or subsequent MESSAGE_UPDATE is dropped.
                                        if event_name == "MESSAGE_CREATE" {
                                            threaded_message_ids
                                                .write()
                                                .await
                                                .insert(msg.platform_message_id.clone());
                                        }

                                        if tx.send(msg).await.is_err() {
                                            return;
                                        }
                                    }
                                }

                                "THREAD_DELETE" => {
                                    if let Some(tid) = d["id"].as_str() {
                                        created_thread_ids.write().await.remove(tid);
                                        let mut ids = threaded_message_ids.write().await;
                                        if ids.len() > MAX_DEDUP_MSG_IDS {
                                            ids.clear();
                                        }
                                        debug!("Discord thread deleted: {tid}");
                                    }
                                }

                                "CHANNEL_DELETE" => {
                                    if let Some(cid) = d["id"].as_str() {
                                        created_thread_ids.write().await.remove(cid);
                                        let mut ids = threaded_message_ids.write().await;
                                        if ids.len() > MAX_DEDUP_MSG_IDS {
                                            ids.clear();
                                        }
                                        debug!("Discord channel deleted: {cid}");
                                    }
                                    // Evict from the (guild_id, name) → channel_id
                                    // cache so ANAI-55 `#name` resolution doesn't keep
                                    // routing to a dead channel.
                                    if let (Some(gid), Some(name)) =
                                        (d["guild_id"].as_str(), d["name"].as_str())
                                    {
                                        channel_name_to_channel_id
                                            .write()
                                            .await
                                            .remove(&(gid.to_string(), name.to_string()));
                                    }
                                }

                                "GUILD_CREATE" | "GUILD_UPDATE" => {
                                    // Populate (guild_id, channel_name) → channel_id from
                                    // the GUILD payload. Requires the GUILDS gateway
                                    // intent (bit 0); default Discord adapter intents
                                    // include it post-ANAI-55.
                                    if let (Some(gid), Some(channels)) =
                                        (d["id"].as_str(), d["channels"].as_array())
                                    {
                                        let mut cache = channel_name_to_channel_id.write().await;
                                        for ch in channels {
                                            if let (Some(cname), Some(cid)) =
                                                (ch["name"].as_str(), ch["id"].as_str())
                                            {
                                                cache.insert(
                                                    (gid.to_string(), cname.to_string()),
                                                    cid.to_string(),
                                                );
                                            }
                                        }
                                    }
                                }

                                "CHANNEL_CREATE" | "CHANNEL_UPDATE" => {
                                    // Keep `(guild_id, name) → channel_id` fresh as
                                    // channels are added or renamed.
                                    if let (Some(gid), Some(cname), Some(cid)) = (
                                        d["guild_id"].as_str(),
                                        d["name"].as_str(),
                                        d["id"].as_str(),
                                    ) {
                                        channel_name_to_channel_id.write().await.insert(
                                            (gid.to_string(), cname.to_string()),
                                            cid.to_string(),
                                        );
                                    }
                                }

                                "RESUMED" => {
                                    info!("Discord session resumed successfully");
                                }

                                _ => {
                                    debug!("Discord event: {event_name}");
                                }
                            }
                        }

                        opcode::HEARTBEAT => {
                            // Server requests immediate heartbeat
                            let seq = *sequence.read().await;
                            let hb = build_heartbeat_payload(seq);
                            let _ = ws_tx
                                .lock()
                                .await
                                .send(tokio_tungstenite::tungstenite::Message::Text(
                                    serde_json::to_string(&hb).unwrap(),
                                ))
                                .await;
                            // The server-requested heartbeat counts as a fresh
                            // beat — reset the ACK gate so the periodic task
                            // doesn't see a stale "unacked" flag.
                            heartbeat_acked.store(false, Ordering::Relaxed);
                        }

                        opcode::HEARTBEAT_ACK => {
                            debug!("Discord heartbeat ACK received");
                            heartbeat_acked.store(true, Ordering::Relaxed);
                        }

                        opcode::RECONNECT => {
                            info!("Discord: server requested reconnect");
                            break 'inner true;
                        }

                        opcode::INVALID_SESSION => {
                            let resumable = payload["d"].as_bool().unwrap_or(false);
                            if resumable {
                                info!("Discord: invalid session (resumable)");
                            } else {
                                info!("Discord: invalid session (not resumable), clearing session");
                                *session_id_store.write().await = None;
                                *sequence.write().await = None;
                            }
                            break 'inner true;
                        }

                        _ => {
                            debug!("Discord: unknown opcode {op}");
                        }
                    }
                };

                // Tear down the heartbeat task before we either exit or
                // reconnect, so it doesn't outlive its WebSocket sink.
                if let Some(h) = heartbeat_handle.take() {
                    h.abort();
                }

                if !should_reconnect || *shutdown.borrow() {
                    break;
                }

                // Try resume URL if available
                if let Some(ref url) = *resume_url_store.read().await {
                    connect_url = format!("{url}/?v=10&encoding=json");
                }

                warn!("Discord: reconnecting in {backoff:?}");
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }

            info!("Discord gateway loop stopped");
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    async fn send(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // platform_id is the channel_id for Discord
        let channel_id = &user.platform_id;
        match content {
            ChannelContent::Text(text) => {
                self.api_send_message(channel_id, &text).await?;
            }
            _ => {
                self.api_send_message(channel_id, "(Unsupported content type)")
                    .await?;
            }
        }
        Ok(())
    }

    async fn send_typing(&self, user: &ChannelUser) -> Result<(), Box<dyn std::error::Error>> {
        self.api_send_typing(&user.platform_id).await
    }

    async fn send_in_thread(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
        thread_id: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let channel_id = &user.platform_id;
        match content {
            ChannelContent::Text(text) => {
                self.api_send_thread_message(channel_id, thread_id, &text)
                    .await?;
            }
            _ => {
                self.api_send_thread_message(channel_id, thread_id, "(Unsupported content type)")
                    .await?;
            }
        }
        Ok(())
    }

    async fn create_thread(
        &self,
        user: &ChannelUser,
        message_id: &str,
        thread_name: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let channel_id = &user.platform_id;
        let thread_id = self
            .api_create_thread(channel_id, message_id, thread_name)
            .await?;
        // Also ensure the message_id is marked as seen (belt-and-suspenders:
        // the gateway loop already inserts on MESSAGE_CREATE, but keep this
        // in case create_thread is ever called from another path).
        self.threaded_message_ids
            .write()
            .await
            .insert(message_id.to_string());
        Ok(thread_id)
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.shutdown_tx.send(true);
        Ok(())
    }

    /// Discord recipient resolver (ANAI-55).
    ///
    /// Resolution order: snowflake → `<#…>` channel mention → `<@…>` user
    /// mention → `#name` / single-token channel name → `@username` → bare
    /// name (refused). See `2026-05-28-anai-55-proposal-v2.md` §3.2.
    async fn resolve_recipient(&self, recipient: &str) -> Result<ChannelUser, ResolutionError> {
        match self.resolve_recipient_inner(recipient).await {
            Ok((platform_id, via)) => {
                // Dedicated audit sink: bypasses tracing entirely so a log
                // scrape on stderr/tui.log cannot correlate this event with
                // adjacent payload logs by timestamp (ANAI-55 security ask).
                crate::resolver_audit::record_resolution("discord", recipient, &platform_id, via);
                // Demoted from info! to debug! — production EnvFilter (info)
                // emits zero correlatable event to the main log; dev mode
                // opts in via RUST_LOG=...resolver_audit=debug.
                #[cfg(debug_assertions)]
                debug!(
                    target: RESOLVER_AUDIT_TARGET,
                    "recipient_resolved adapter=discord input={:?} resolved_platform_id={} via={}",
                    recipient, platform_id, via
                );
                Ok(ChannelUser {
                    platform_id,
                    display_name: recipient.to_string(),
                    openfang_user: None,
                })
            }
            Err(e) => {
                if self.note_resolution_failure().await {
                    warn!(
                        "discord resolver: >{} failures in last {}s (most recent: {})",
                        RESOLVER_FAIL_WARN_THRESHOLD,
                        RESOLVER_FAIL_WINDOW.as_secs(),
                        e
                    );
                }
                Err(e)
            }
        }
    }
}

/// Maximum byte size for an attachment to be classified as a vision-eligible
/// image. Anthropic's image content blocks are capped at 5 MB; oversize images
/// fall through to `File` so the bridge passes the URL as text instead of
/// attempting an inline image block.
const VISION_IMAGE_MAX_BYTES: u64 = 5 * 1024 * 1024;

/// Best-effort MIME inference from a filename extension. Used as a fallback
/// when Discord's `content_type` field is missing or empty (we've observed
/// this on some bot-relayed attachments).
fn mime_from_extension(filename: &str) -> Option<&'static str> {
    let ext = filename.rsplit('.').next()?.to_ascii_lowercase();
    match ext.as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "heic" => Some("image/heic"),
        "heif" => Some("image/heif"),
        "pdf" => Some("application/pdf"),
        "txt" => Some("text/plain"),
        "md" => Some("text/markdown"),
        "json" => Some("application/json"),
        "mp4" => Some("video/mp4"),
        "mov" => Some("video/quicktime"),
        "mp3" => Some("audio/mpeg"),
        "wav" => Some("audio/wav"),
        "ogg" => Some("audio/ogg"),
        _ => None,
    }
}

/// Classify a single Discord attachment JSON object into a `ChannelContent`
/// block. Vision-eligible image MIME types (jpeg/png/gif/webp) under
/// `VISION_IMAGE_MAX_BYTES` become `Image`; everything else becomes `File`
/// (URL-pass-through; the bridge will surface it as a text descriptor in v1).
///
/// MIME resolution chain: `attachments[].content_type` (if non-empty) →
/// extension lookup → `application/octet-stream`.
fn classify_discord_attachment(att: &serde_json::Value) -> ChannelContent {
    let url = att["url"].as_str().unwrap_or("").to_string();
    let filename = att["filename"].as_str().unwrap_or("file").to_string();
    let size = att["size"].as_u64();

    let resolved_mime: String = att["content_type"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| mime_from_extension(&filename).map(str::to_string))
        .unwrap_or_else(|| "application/octet-stream".to_string());

    let is_vision_mime = matches!(
        resolved_mime.as_str(),
        "image/jpeg" | "image/png" | "image/gif" | "image/webp"
    );
    // If size is unknown, optimistically allow the image — the bridge will
    // surface a 4xx if Anthropic rejects it, which is better than silently
    // demoting to a text URL.
    let within_vision_limit = size.map(|s| s <= VISION_IMAGE_MAX_BYTES).unwrap_or(true);

    if is_vision_mime && within_vision_limit {
        ChannelContent::Image { url, caption: None }
    } else {
        ChannelContent::File {
            url,
            filename,
            mime: Some(resolved_mime),
            size,
        }
    }
}

/// True for ASCII-digit strings 17-20 chars long (Discord snowflake shape).
fn is_snowflake(s: &str) -> bool {
    let len = s.len();
    (17..=20).contains(&len) && s.bytes().all(|b| b.is_ascii_digit())
}

/// Match `<#1234567890>` and return the inner digits.
fn strip_channel_mention(s: &str) -> Option<String> {
    let inner = s.strip_prefix("<#")?.strip_suffix('>')?;
    if !inner.is_empty() && inner.bytes().all(|b| b.is_ascii_digit()) {
        Some(inner.to_string())
    } else {
        None
    }
}

/// Match `<@1234567890>` or `<@!1234567890>` and return the inner digits.
fn strip_user_mention(s: &str) -> Option<String> {
    let inner = s.strip_prefix("<@")?.strip_suffix('>')?;
    let inner = inner.strip_prefix('!').unwrap_or(inner);
    if !inner.is_empty() && inner.bytes().all(|b| b.is_ascii_digit()) {
        Some(inner.to_string())
    } else {
        None
    }
}

/// Discord channel-name charset: lowercase alphanumeric, `-`, `_`.
fn is_channel_name_shape(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

/// Discord username charset (post-discriminator world): lowercase
/// alphanumeric, `.`, `_`.
fn is_username_shape(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.' || b == b'_')
}

/// Parse a Discord MESSAGE_CREATE or MESSAGE_UPDATE payload into a `ChannelMessage`.
async fn parse_discord_message(
    d: &serde_json::Value,
    bot_user_id: &Arc<RwLock<Option<String>>>,
    allowed_guilds: &[String],
    allowed_users: &[String],
    ignore_bots: bool,
    created_thread_ids: &Arc<RwLock<HashMap<String, String>>>,
) -> Option<ChannelMessage> {
    // Diagnostic: dump the raw Discord payload so we can ground attachment
    // parsing in real JSON. Gated by RUST_LOG; silent at default `info` level.
    // Enable with: RUST_LOG=openfang_channels::discord=debug
    debug!(target: "openfang_channels::discord", payload = %d, "discord raw message payload");

    let author = d.get("author")?;
    let author_id = author["id"].as_str()?;

    // Filter out bot's own messages
    if let Some(ref bid) = *bot_user_id.read().await {
        if author_id == bid {
            return None;
        }
    }

    // Filter out other bots (configurable via ignore_bots)
    if ignore_bots && author["bot"].as_bool() == Some(true) {
        return None;
    }

    // Filter by allowed users
    if !allowed_users.is_empty() && !allowed_users.iter().any(|u| u == author_id) {
        debug!("Discord: ignoring message from unlisted user {author_id}");
        return None;
    }

    // Filter by allowed guilds
    if !allowed_guilds.is_empty() {
        if let Some(guild_id) = d["guild_id"].as_str() {
            if !allowed_guilds.iter().any(|g| g == guild_id) {
                return None;
            }
        }
    }

    let content_text = d["content"].as_str().unwrap_or("");
    let channel_id = d["channel_id"].as_str()?;
    let message_id = d["id"].as_str().unwrap_or("0");

    // Detect if this message is inside a bot-created thread.
    // In Discord, a thread is its own channel — channel_id will be the thread's ID.
    // If so, use the parent channel as platform_id and set thread_id so that:
    //  (a) auto-thread logic is skipped (message.thread_id.is_some())
    //  (b) responses are sent back into the same thread
    let (effective_channel_id, parsed_thread_id) = {
        let threads = created_thread_ids.read().await;
        if let Some(parent_channel_id) = threads.get(channel_id) {
            (parent_channel_id.clone(), Some(channel_id.to_string()))
        } else {
            (channel_id.to_string(), None)
        }
    };
    let username = author["username"].as_str().unwrap_or("Unknown");
    let discriminator = author["discriminator"].as_str().unwrap_or("0000");
    let display_name = if discriminator == "0" {
        username.to_string()
    } else {
        format!("{username}#{discriminator}")
    };

    let timestamp = d["timestamp"]
        .as_str()
        .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(chrono::Utc::now);

    // Parse commands (messages starting with /). Commands do not carry
    // attachments in v1; attachment processing only runs in the non-command path.
    let content = if content_text.starts_with('/') {
        let parts: Vec<&str> = content_text.splitn(2, ' ').collect();
        let cmd_name = &parts[0][1..];
        let args = if parts.len() > 1 {
            parts[1].split_whitespace().map(String::from).collect()
        } else {
            vec![]
        };
        ChannelContent::Command {
            name: cmd_name.to_string(),
            args,
        }
    } else {
        let attachment_blocks: Vec<ChannelContent> = d["attachments"]
            .as_array()
            .map(|arr| arr.iter().map(classify_discord_attachment).collect())
            .unwrap_or_default();

        match (content_text.is_empty(), attachment_blocks.len()) {
            // No text, no attachments → nothing to ingest.
            (true, 0) => return None,
            // Text only.
            (false, 0) => ChannelContent::Text(content_text.to_string()),
            // Single attachment, no caption.
            (true, 1) => attachment_blocks.into_iter().next().unwrap(),
            // Single attachment + caption: emit Multipart with the caption as
            // a sibling Text block. This keeps the caption visible to providers
            // that flatten content to text only (e.g. claude-code/*, which
            // currently drops Image blocks) — the user gets a coherent
            // text-only response instead of a hallucination. Vision-capable
            // providers see the same blocks and dispatch multimodally.
            (false, 1) => {
                let block = attachment_blocks.into_iter().next().unwrap();
                let normalized = match block {
                    // Drop any caption that classify_discord_attachment may have
                    // attached; the sibling Text block is now the caption.
                    ChannelContent::Image { url, caption: _ } => {
                        ChannelContent::Image { url, caption: None }
                    }
                    other => other,
                };
                ChannelContent::Multipart(vec![
                    ChannelContent::Text(content_text.to_string()),
                    normalized,
                ])
            }
            // Multiple attachments, no caption.
            (true, _) => ChannelContent::Multipart(attachment_blocks),
            // Multiple attachments + caption: text first, then attachments
            // (matches Discord's visual ordering: text above attachments).
            (false, _) => {
                let mut blocks = Vec::with_capacity(attachment_blocks.len() + 1);
                blocks.push(ChannelContent::Text(content_text.to_string()));
                blocks.extend(attachment_blocks);
                ChannelContent::Multipart(blocks)
            }
        }
    };

    // Determine if this is a group message (guild_id present = server channel)
    let is_group = d["guild_id"].as_str().is_some();

    // Check if bot was @mentioned (for MentionOnly policy enforcement)
    let was_mentioned = if let Some(ref bid) = *bot_user_id.read().await {
        // Check Discord mentions array
        let mentioned_in_array = d["mentions"]
            .as_array()
            .map(|arr| arr.iter().any(|m| m["id"].as_str() == Some(bid.as_str())))
            .unwrap_or(false);
        // Also check content for <@bot_id> or <@!bot_id> patterns
        let mentioned_in_content = content_text.contains(&format!("<@{bid}>"))
            || content_text.contains(&format!("<@!{bid}>"));
        mentioned_in_array || mentioned_in_content
    } else {
        false
    };

    let mut metadata = HashMap::new();
    if was_mentioned {
        metadata.insert("was_mentioned".to_string(), serde_json::json!(true));
    }
    // Stash the Discord author ID so the router can key bindings on user, not channel.
    // (`sender.platform_id` below is the channel ID, used for the send path.)
    metadata.insert("sender_user_id".to_string(), serde_json::json!(author_id));

    Some(ChannelMessage {
        channel: ChannelType::Discord,
        platform_message_id: message_id.to_string(),
        sender: ChannelUser {
            platform_id: effective_channel_id,
            display_name,
            openfang_user: None,
        },
        content,
        target_agent: None,
        timestamp,
        is_group,
        thread_id: parsed_thread_id,
        metadata,
    })
}

/// Build a Discord thread name from the message content.
/// Strips @mention prefixes (`<@...>`), trims whitespace, and truncates to
/// Discord's 100-character thread name limit. Falls back to the sender's
/// display name if the message has no usable text (e.g. image-only).
fn thread_name_from_message(message: &ChannelMessage) -> String {
    let raw = match &message.content {
        ChannelContent::Text(t) => t.clone(),
        ChannelContent::Image { caption, .. } => caption.clone().unwrap_or_default(),
        _ => String::new(),
    };

    // Strip leading Discord mention tokens (<@id> / <@!id>)
    let stripped = regex_lite::Regex::new(r"^(<@!?\d+>\s*)+")
        .map(|re| re.replace(&raw, "").into_owned())
        .unwrap_or(raw);

    let trimmed = stripped.trim().to_string();

    if trimmed.is_empty() {
        return message.sender.display_name.clone();
    }

    // Truncate to Discord's 100-char limit
    if trimmed.chars().count() <= 100 {
        trimmed
    } else {
        trimmed.chars().take(97).collect::<String>() + "…"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience helper: empty thread-tracking map for tests that don't exercise threading.
    fn empty_threads() -> Arc<RwLock<HashMap<String, String>>> {
        Arc::new(RwLock::new(HashMap::new()))
    }

    #[tokio::test]
    async fn test_parse_discord_message_basic() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Hello agent!",
            "author": {
                "id": "user456",
                "username": "alice",
                "discriminator": "0",
                "bot": false
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        assert_eq!(msg.channel, ChannelType::Discord);
        assert_eq!(msg.sender.display_name, "alice");
        assert_eq!(msg.sender.platform_id, "ch1");
        assert!(matches!(msg.content, ChannelContent::Text(ref t) if t == "Hello agent!"));
    }

    #[tokio::test]
    async fn test_parse_discord_message_filters_bot() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "My own message",
            "author": {
                "id": "bot123",
                "username": "openfang",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads()).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_discord_message_filters_other_bots() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Bot message",
            "author": {
                "id": "other_bot",
                "username": "somebot",
                "discriminator": "0",
                "bot": true
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads()).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_discord_ignore_bots_false_allows_other_bots() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Bot message",
            "author": {
                "id": "other_bot",
                "username": "somebot",
                "discriminator": "0",
                "bot": true
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        // With ignore_bots=false, other bots' messages should be allowed
        let msg = parse_discord_message(&d, &bot_id, &[], &[], false, &empty_threads()).await;
        assert!(msg.is_some());
        let msg = msg.unwrap();
        assert_eq!(msg.sender.display_name, "somebot");
        assert!(matches!(msg.content, ChannelContent::Text(ref t) if t == "Bot message"));
    }

    #[tokio::test]
    async fn test_parse_discord_ignore_bots_false_still_filters_self() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "My own message",
            "author": {
                "id": "bot123",
                "username": "openfang",
                "discriminator": "0",
                "bot": true
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        // Even with ignore_bots=false, the bot's own messages must still be filtered
        let msg = parse_discord_message(&d, &bot_id, &[], &[], false, &empty_threads()).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_discord_message_guild_filter() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "guild_id": "999",
            "content": "Hello",
            "author": {
                "id": "user1",
                "username": "bob",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        // Not in allowed guilds
        let msg = parse_discord_message(
            &d,
            &bot_id,
            &["111".into(), "222".into()],
            &[],
            true,
            &empty_threads(),
        )
        .await;
        assert!(msg.is_none());

        // In allowed guilds
        let msg =
            parse_discord_message(&d, &bot_id, &["999".into()], &[], true, &empty_threads()).await;
        assert!(msg.is_some());
    }

    #[tokio::test]
    async fn test_parse_discord_command() {
        let bot_id = Arc::new(RwLock::new(None));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "/agent hello-world",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        match &msg.content {
            ChannelContent::Command { name, args } => {
                assert_eq!(name, "agent");
                assert_eq!(args, &["hello-world"]);
            }
            other => panic!("Expected Command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_discord_empty_content() {
        let bot_id = Arc::new(RwLock::new(None));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads()).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_discord_discriminator() {
        let bot_id = Arc::new(RwLock::new(None));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Hi",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "1234"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        assert_eq!(msg.sender.display_name, "alice#1234");
    }

    #[tokio::test]
    async fn test_parse_discord_message_update() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Edited message content",
            "author": {
                "id": "user456",
                "username": "alice",
                "discriminator": "0",
                "bot": false
            },
            "timestamp": "2024-01-01T00:00:00+00:00",
            "edited_timestamp": "2024-01-01T00:01:00+00:00"
        });

        // MESSAGE_UPDATE uses the same parse function as MESSAGE_CREATE
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        assert_eq!(msg.channel, ChannelType::Discord);
        assert!(
            matches!(msg.content, ChannelContent::Text(ref t) if t == "Edited message content")
        );
    }

    #[tokio::test]
    async fn test_parse_discord_allowed_users_filter() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Hello",
            "author": {
                "id": "user999",
                "username": "bob",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        // Not in allowed users
        let msg = parse_discord_message(
            &d,
            &bot_id,
            &[],
            &["user111".into(), "user222".into()],
            true,
            &empty_threads(),
        )
        .await;
        assert!(msg.is_none());

        // In allowed users
        let msg = parse_discord_message(
            &d,
            &bot_id,
            &[],
            &["user999".into()],
            true,
            &empty_threads(),
        )
        .await;
        assert!(msg.is_some());

        // Empty allowed_users = allow all
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads()).await;
        assert!(msg.is_some());
    }

    #[tokio::test]
    async fn test_parse_discord_mention_detection() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));

        // Message with bot mentioned in mentions array
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "guild_id": "guild1",
            "content": "Hey <@bot123> help me",
            "mentions": [{"id": "bot123", "username": "openfang"}],
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        assert!(msg.is_group);
        assert_eq!(
            msg.metadata.get("was_mentioned").and_then(|v| v.as_bool()),
            Some(true)
        );

        // Message without mention in group
        let d2 = serde_json::json!({
            "id": "msg2",
            "channel_id": "ch1",
            "guild_id": "guild1",
            "content": "Just chatting",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg2 = parse_discord_message(&d2, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        assert!(msg2.is_group);
        assert!(!msg2.metadata.contains_key("was_mentioned"));
    }

    #[tokio::test]
    async fn test_parse_discord_dm_not_group() {
        let bot_id = Arc::new(RwLock::new(None));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "dm-ch1",
            "content": "Hello",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        assert!(!msg.is_group);
    }

    #[test]
    fn test_build_heartbeat_payload_with_sequence() {
        let payload = build_heartbeat_payload(Some(42));
        assert_eq!(payload["op"], 1);
        assert_eq!(payload["d"], 42);
        // Round-trip through serde_json::to_string and re-parse to assert
        // valid JSON matching {"op":1,"d":42} regardless of key ordering.
        let s = serde_json::to_string(&payload).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, serde_json::json!({"op": 1, "d": 42}));
    }

    #[test]
    fn test_build_heartbeat_payload_without_sequence() {
        let payload = build_heartbeat_payload(None);
        assert_eq!(payload["op"], 1);
        assert!(payload["d"].is_null());
        let s = serde_json::to_string(&payload).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({"op": 1, "d": serde_json::Value::Null})
        );
    }

    #[test]
    fn test_discord_adapter_creation() {
        let adapter = DiscordAdapter::new(
            "test-token".to_string(),
            vec!["123".to_string(), "456".to_string()],
            vec![],
            true,
            37376,
            "true".to_string(),
        );
        assert_eq!(adapter.name(), "discord");
        assert_eq!(adapter.channel_type(), ChannelType::Discord);
    }

    // -- Multipart / attachment parsing tests (commit 4) ----------------------

    fn att(filename: &str, content_type: Option<&str>, size: u64) -> serde_json::Value {
        let mut obj = serde_json::json!({
            "url": format!("https://cdn.discordapp.com/attachments/1/2/{filename}"),
            "filename": filename,
            "size": size,
        });
        if let Some(ct) = content_type {
            obj["content_type"] = serde_json::Value::String(ct.to_string());
        }
        obj
    }

    fn payload_with(content: &str, attachments: Vec<serde_json::Value>) -> serde_json::Value {
        serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": content,
            "author": {
                "id": "user456",
                "username": "alice",
                "discriminator": "0",
                "bot": false
            },
            "timestamp": "2024-01-01T00:00:00+00:00",
            "attachments": attachments,
        })
    }

    #[tokio::test]
    async fn test_parse_image_only_no_caption() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = payload_with("", vec![att("photo.png", Some("image/png"), 100_000)]);
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        match msg.content {
            ChannelContent::Image { caption, url } => {
                assert!(caption.is_none());
                assert!(url.contains("photo.png"));
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_image_with_caption() {
        // Single image + caption is emitted as Multipart([Text, Image]) so the
        // caption survives providers that flatten content blocks to text only
        // (e.g. claude-code/*). The Image carries no caption of its own; the
        // sibling Text block IS the caption.
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = payload_with(
            "look at this",
            vec![att("photo.jpg", Some("image/jpeg"), 50_000)],
        );
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        match msg.content {
            ChannelContent::Multipart(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(&parts[0], ChannelContent::Text(t) if t == "look at this"));
                match &parts[1] {
                    ChannelContent::Image { caption, url } => {
                        assert!(
                            caption.is_none(),
                            "image caption should be None; the sibling Text block is the caption"
                        );
                        assert!(url.contains("photo.jpg"));
                    }
                    other => panic!("expected Image as second part, got {other:?}"),
                }
            }
            other => panic!("expected Multipart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_multi_image_no_caption() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = payload_with(
            "",
            vec![
                att("a.png", Some("image/png"), 10_000),
                att("b.png", Some("image/png"), 20_000),
            ],
        );
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        match msg.content {
            ChannelContent::Multipart(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(parts
                    .iter()
                    .all(|p| matches!(p, ChannelContent::Image { .. })));
            }
            other => panic!("expected Multipart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_multi_image_with_caption() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = payload_with(
            "two pics",
            vec![
                att("a.png", Some("image/png"), 10_000),
                att("b.png", Some("image/png"), 20_000),
            ],
        );
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        match msg.content {
            ChannelContent::Multipart(parts) => {
                assert_eq!(parts.len(), 3);
                // Text first, then images.
                assert!(matches!(&parts[0], ChannelContent::Text(t) if t == "two pics"));
                assert!(matches!(&parts[1], ChannelContent::Image { .. }));
                assert!(matches!(&parts[2], ChannelContent::Image { .. }));
            }
            other => panic!("expected Multipart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_heic_falls_to_file() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = payload_with("", vec![att("photo.heic", Some("image/heic"), 100_000)]);
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        match msg.content {
            ChannelContent::File { mime, filename, .. } => {
                assert_eq!(filename, "photo.heic");
                assert_eq!(mime.as_deref(), Some("image/heic"));
            }
            other => panic!("expected File, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_oversize_image_falls_to_file() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        // 6 MB exceeds VISION_IMAGE_MAX_BYTES (5 MB).
        let d = payload_with(
            "",
            vec![att("huge.png", Some("image/png"), 6 * 1024 * 1024)],
        );
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        match msg.content {
            ChannelContent::File {
                filename,
                mime,
                size,
                ..
            } => {
                assert_eq!(filename, "huge.png");
                assert_eq!(mime.as_deref(), Some("image/png"));
                assert_eq!(size, Some(6 * 1024 * 1024));
            }
            other => panic!("expected File, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_file_with_caption_yields_multipart() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = payload_with(
            "see attached",
            vec![att("doc.pdf", Some("application/pdf"), 200_000)],
        );
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        match msg.content {
            ChannelContent::Multipart(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(&parts[0], ChannelContent::Text(t) if t == "see attached"));
                assert!(matches!(&parts[1], ChannelContent::File { .. }));
            }
            other => panic!("expected Multipart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_extension_fallback_when_content_type_missing() {
        // Discord occasionally omits content_type on bot-relayed attachments;
        // we should fall back to the filename extension.
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = payload_with("", vec![att("pic.png", None, 50_000)]);
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads())
            .await
            .unwrap();
        assert!(matches!(msg.content, ChannelContent::Image { .. }));
    }

    #[tokio::test]
    async fn test_parse_empty_message_with_no_attachments_returns_none() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = payload_with("", vec![]);
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true, &empty_threads()).await;
        assert!(msg.is_none());
    }

    // --- ANAI-55 resolver tests --------------------------------------------

    fn make_adapter() -> DiscordAdapter {
        DiscordAdapter::new(
            "test_token".to_string(),
            vec![],
            vec![],
            true,
            37377,
            "false".to_string(),
        )
    }

    #[tokio::test]
    async fn resolver_passthrough_snowflake() {
        let a = make_adapter();
        let u = a.resolve_recipient("1476626559401066688").await.unwrap();
        assert_eq!(u.platform_id, "1476626559401066688");
    }

    #[tokio::test]
    async fn resolver_channel_mention() {
        let a = make_adapter();
        let u = a.resolve_recipient("<#1476626559401066688>").await.unwrap();
        assert_eq!(u.platform_id, "1476626559401066688");
    }

    #[tokio::test]
    async fn resolver_user_mention_uncached_attempts_dm_open() {
        // No DM cache entry → resolver falls through to api_open_dm_channel,
        // which fails (no network) and we get DmOpenFailed. We only assert
        // the *error class*, not the status code, since transport failure
        // surfaces as `0`.
        let a = make_adapter();
        let err = a
            .resolve_recipient("<@1086446153098342510>")
            .await
            .unwrap_err();
        assert!(
            matches!(err, ResolutionError::DmOpenFailed { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn resolver_channel_name_single_guild() {
        let a = make_adapter();
        a.channel_name_to_channel_id.write().await.insert(
            ("guild_A".to_string(), "system-code".to_string()),
            "1476626559401066688".to_string(),
        );
        let u = a.resolve_recipient("#system-code").await.unwrap();
        assert_eq!(u.platform_id, "1476626559401066688");

        let u2 = a.resolve_recipient("system-code").await.unwrap();
        assert_eq!(u2.platform_id, "1476626559401066688");
    }

    #[tokio::test]
    async fn resolver_channel_name_ambiguous_across_guilds() {
        let a = make_adapter();
        {
            let mut c = a.channel_name_to_channel_id.write().await;
            c.insert(
                ("guild_A".to_string(), "general".to_string()),
                "111".to_string(),
            );
            c.insert(
                ("guild_B".to_string(), "general".to_string()),
                "222".to_string(),
            );
        }
        let err = a.resolve_recipient("#general").await.unwrap_err();
        match err {
            ResolutionError::AmbiguousChannel { name, guilds } => {
                assert_eq!(name, "general");
                assert_eq!(guilds.len(), 2);
            }
            other => panic!("expected AmbiguousChannel, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn resolver_explicit_hash_no_hit_is_unknown() {
        let a = make_adapter();
        let err = a.resolve_recipient("#nope").await.unwrap_err();
        assert!(matches!(err, ResolutionError::UnknownRecipient { .. }));
    }

    #[tokio::test]
    async fn resolver_bare_name_with_no_channel_is_refused_as_dm() {
        let a = make_adapter();
        let err = a.resolve_recipient("benhoverter").await.unwrap_err();
        match err {
            ResolutionError::BareNameDmRefused { name } => assert_eq!(name, "benhoverter"),
            other => panic!("expected BareNameDmRefused, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn resolver_at_username_uncached_is_unknown() {
        let a = make_adapter();
        let err = a.resolve_recipient("@nobody").await.unwrap_err();
        assert!(matches!(err, ResolutionError::UnknownRecipient { .. }));
    }

    #[tokio::test]
    async fn resolver_at_username_cached_attempts_dm_open() {
        let a = make_adapter();
        a.username_to_user_id
            .write()
            .await
            .insert("benhoverter".to_string(), "1086446153098342510".to_string());
        let err = a.resolve_recipient("@benhoverter").await.unwrap_err();
        assert!(matches!(err, ResolutionError::DmOpenFailed { .. }));
    }

    #[tokio::test]
    async fn resolver_dm_channel_cache_hit_skips_api_call() {
        let a = make_adapter();
        a.username_to_user_id
            .write()
            .await
            .insert("benhoverter".to_string(), "1086446153098342510".to_string());
        a.user_id_to_dm_channel_id
            .write()
            .await
            .insert("1086446153098342510".to_string(), "dm_chan_999".to_string());
        let u = a.resolve_recipient("@benhoverter").await.unwrap();
        assert_eq!(u.platform_id, "dm_chan_999");
    }

    #[tokio::test]
    async fn resolver_empty_input_is_unknown() {
        let a = make_adapter();
        let err = a.resolve_recipient("").await.unwrap_err();
        assert!(matches!(err, ResolutionError::UnknownRecipient { .. }));
    }
}
