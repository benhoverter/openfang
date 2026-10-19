# OpenFang (fork) quickstart: macOS + Claude Code + Discord

This gets you from nothing to an agent answering you in Discord, running on your Claude Code subscription. Plan on about an hour, most of it waiting for the first build.

> **This is a fork.** The docs at openfang.sh and the install commands there (the curl installer, Docker, `cargo install`) install *upstream* OpenFang, which has none of what's described here. Follow this page instead.

**What you get:** a background service (the *daemon*) that runs your agents, each one a Claude Code process with OpenFang's tools (files, memory, web, shell under an allowlist), reachable from Discord channels you choose.

---

## 1. Prerequisites

| need | check | install |
|---|---|---|
| macOS 14 or later | `sw_vers` | |
| Xcode command-line tools | `xcode-select -p` | `xcode-select --install` |
| Rust | `cargo --version` | <https://rustup.rs> |
| Claude Code, logged in | `claude --version`, then `claude` and `/login` once | <https://docs.claude.com/en/docs/claude-code> |
| A Discord server you control | | |

Claude Code must be logged in under your macOS user. The daemon runs `claude` as you and uses your subscription; there's no API key to set.

## 2. Get the code and build it

```sh
git clone https://github.com/benhoverter/openfang.git ~/openfang
cd ~/openfang
cargo build --release --bin openfang --bin openfang-mcp-bridge
```

The first build takes 10–30 minutes. Then put both programs where the service expects them:

```sh
mkdir -p ~/.openfang/bin
cp target/release/openfang target/release/openfang-mcp-bridge ~/.openfang/bin/
echo 'export PATH="$HOME/.openfang/bin:$PATH"' >> ~/.zshrc
exec zsh
```

`openfang-mcp-bridge` is the piece that gives Claude Code agents OpenFang's tools. Without it, agents start but can't do anything, so both files have to be there.

## 3. Create the Discord bot

1. Go to <https://discord.com/developers/applications> → **New Application**. Name it after your agents.
2. **Bot** tab → **Reset Token** → copy the token. You'll paste it in step 4. Treat it like a password.
3. Same tab, under **Privileged Gateway Intents**, turn on **Message Content Intent**. Without it the bot sees messages arrive but can't read them.
4. **OAuth2 → URL Generator**: scopes `bot`; permissions *View Channels, Send Messages, Send Messages in Threads, Read Message History, Attach Files, Add Reactions*. Open the URL it gives you and add the bot to your server.
5. In Discord: **User Settings → Advanced → Developer Mode** on. Now you can right-click a channel or yourself and **Copy ID**.

Create a channel for your first agent (say `#assistant`) and copy its ID. Copy your own user ID too.

## 4. Store the bot token

```sh
mkdir -p ~/.openfang
printf 'DISCORD_BOT_TOKEN=%s\n' 'paste-token-here' >> ~/.openfang/secrets.env
chmod 600 ~/.openfang/secrets.env
```

The daemon reads `secrets.env` itself every time it starts. Secrets never go in `config.toml` or in the service file, and agents can't read this file.

## 5. Install the starter agents

```sh
openfang init --starter
```

It asks a few questions. Or pass them as flags:

```sh
openfang init --starter \
  --roles assistant,guide \
  --user-name "Your Name" \
  --channel assistant=<#assistant channel ID> \
  --discord-user <your user ID>
```

**The roles:**

| role | what it's for | shell |
|---|---|---|
| `assistant` | general help; your first agent | inspection only (`ls`, `cat`, `rg`…) |
| `guide` | answers "how do I…?" about this install; read-only | inspection only |
| `researcher` | web research with sources | none |
| `coder` | works in one repo (`--repo ~/code/myproject`) | `git` and dev tools, allowlisted |
| `alpha` | lead agent that coordinates the others; needs two or more other roles | `git` and dev tools, allowlisted |

Start with `assistant,guide`. You can re-run with more roles later; existing files are never overwritten.

**What it writes:**
- `~/.openfang/config.toml`: Claude Code as the model provider, a safe global shell list, and the Discord section with your channel binding. If you already have a config, it prints the lines to add instead.
- `~/.openfang/RULES.md`: the operating rules every agent follows. Yours to edit.
- `~/.openfang/agents/<role>/agent.toml`: each agent's permissions and system prompt.
- `~/.openfang/workspaces/<role>/`: each agent's persona files (`SOUL.md` voice, `IDENTITY.md`, `AGENTS.md` habits, `USER.md`, `MEMORY.md`). All plain markdown, all yours to edit.

All your agents share one name and voice (default **Annabelle**; change it with `--persona`). Each role is the same person wearing a different hat.

## 6. Start the service

```sh
openfang service install
```

This writes the launchd job (macOS's service manager) and starts the daemon. It keeps the daemon running across crashes and logins. It refuses to run if `openfang-mcp-bridge` is missing from `~/.openfang/bin`, and warns if it can't find `claude` on your PATH; run it from the same terminal where `claude --version` works.

Check it:

```sh
openfang service status
tail -f ~/.openfang/daemon.stderr.log
```

## 7. Say hello

Post in your `#assistant` channel. The first reply takes a few seconds while Claude Code starts. On its first conversation the agent introduces itself and asks a question or two about you.

If nothing comes back, see [Troubleshooting](#troubleshooting).

---

## Everyday use

**Bind another channel to an agent.** Create the channel, copy its ID, then add to `~/.openfang/config.toml`:

```toml
[[bindings]]
agent = "guide"
match_rule = { channel = "discord", channel_id = "123456789012345678" }
```

Add the same ID to `free_response_channels` under `[channels.discord]` so the agent answers without being @mentioned. Then `openfang service restart`.

**Change an agent.** Persona and habits: edit the files in `~/.openfang/workspaces/<agent>/`; they're read fresh every turn. Permissions, model or system prompt: edit `~/.openfang/agents/<agent>/agent.toml`, then `openfang service restart`.

**Give an agent more shell commands.** In its `agent.toml` under `[exec_policy]`: `trusted_commands` run with no prompt; `allowed_commands` run but may ask you first, with ✅/❌ buttons in Discord. An approval that times out means the command did **not** run.

**Service commands.**

| command | does |
|---|---|
| `openfang service status` | is it loaded and running, PID, bridge settings |
| `openfang service restart` | restart; reloads the service file too if it changed |
| `openfang service stop` / `start` | stop it so it stays down / start it again |
| `openfang service uninstall` | remove the service (keeps a backup of its file) |

`openfang stop` on its own just bounces the daemon, because launchd restarts it straight away. Use `service stop` to keep it down.

**Update to the latest fork.**

```sh
cd ~/openfang && git pull
scripts/deploy-local.sh update
```

The deploy script runs the tests, builds, backs up the running binary, swaps in the new one and restarts the service. If the tests fail it stops before touching anything. Rolling back: copy the backup path it printed over `~/.openfang/bin/openfang`, then `openfang service restart`.

---

## Troubleshooting

Ask `guide` first. It can read your agents, workspaces, the daemon log and these docs. It can't read `config.toml` or `secrets.env`; nothing can, by design, so it'll tell you what to check there.

**The bot is online but never answers.**
- Message Content Intent is off (step 3.3).
- The channel isn't bound: no `[[bindings]]` entry for it, and no `default_agent`.
- Your user ID isn't in `allowed_users`.
- The channel isn't in `free_response_channels`, so the bot only answers when @mentioned.
- Search the log: `rg -n "discord|binding" ~/.openfang/daemon.stderr.log | tail -40`.

**The agent answers but says it has no tools, or can't read files.** The MCP bridge isn't wired. `openfang service status` shows the bridge settings; `~/.openfang/bin/openfang-mcp-bridge` must exist and be executable. With the bridge explicitly on (the service default) and the program missing, agents get **no** tools rather than unrestricted ones. That's deliberate: it fails closed. Copy the bridge program back and restart.

**"claude: command not found" in the log.** The service's PATH comes from the shell you ran `service install` in. Run `openfang service install --force` from a terminal where `claude --version` works.

**A command needs approval and nothing happens.** Look for ✅/❌ buttons on the agent's message in Discord. Unanswered prompts time out and the command doesn't run.

**Long commands get killed.** Shell commands time out after 120 seconds for coding agents (60 for the others). Raise `timeout_secs` under `[exec_policy]` in that agent's `agent.toml`.

**I changed `config.toml` or `secrets.env` and nothing happened.** Both are read at startup. `openfang service restart`.

---

## Optional, later

- **Gatekeeper.** An AI reviewer that approves routine gated commands so fewer approval prompts reach you. It ships off. Turn on `shadow = true` under `[gatekeeper]` for a week first: it records what it *would* have approved while every prompt still reaches you. Then decide. A reference page is still to be written; until then the settings are explained in the `GatekeeperConfig` comments in `crates/openfang-types/src/gatekeeper.rs`.
- **`~/CLAUDE.md`.** Claude Code reads this file in every session, including the ones OpenFang starts. If you use Claude Code by hand too, keep anything in it that would contradict your agents' prompts out of it, or add a line saying OpenFang's system prompt wins.
- **More agents.** Copy a starter's `agents/<name>/` and `workspaces/<name>/` folders under a new name, edit, restart.
