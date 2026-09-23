# genai-image-mcp

Zero-dependency Node (>=18) MCP stdio server exposing `generate_image` and
`generate_video` against the Gemini API. Agents see it as
`mcp_genai_image_generate_image` / `mcp_genai_image_generate_video`.

This directory is the **source of truth**. The daemon runs a deployed copy at
`~/.openfang/mcp/genai-image/server.mjs`, registered in `config.toml`.

## Deploy

1. Edit here, run `node smoke.mjs` (offline, no API calls) — must print `ALL PASS`.
2. Copy `server.mjs` over `~/.openfang/mcp/genai-image/server.mjs`
   (keep a `server.mjs.bak.pre-<change>-<date>` backup).
3. Bounce the daemon — MCP servers are spawned once at startup. No rebuild needed.

Never edit the deployed copy directly; it drifts from this one silently.

## Environment

- `GEMINI_API_KEY` — required; read from env only, never a parameter, never logged.
- `HOME` (or `OPENFANG_OUTPUT_ROOT`) — output lands under
  `~/.openfang/workspaces/<agent_name>/output/genai-{images,videos}/`.

## Image models

`model` is optional; an unknown value is a hard error, never a fallback.

| `model` | Model ID | ~Price / image |
|---|---|---|
| `pro` (default) | `gemini-3-pro-image-preview` | 13¢ |
| `flash` | `gemini-3.1-flash-image` | 7¢ |
| `flash-lite` | `gemini-3.1-flash-lite-image` | 3.4¢ (1K only) |

`n` (1–4) makes one API call per image, so it multiplies the price.
