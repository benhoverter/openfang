#!/usr/bin/env node
// genai-image MCP server — vetted in-house, zero-dependency.
// Transport: MCP stdio (newline-delimited JSON-RPC 2.0).
//
// Surface: two tools, `generate_image` and `generate_video`.
//
// AUDIT SURFACE (the whole thing):
//   * ONE network host — GEMINI_HOST below, nothing else. grep for `fetch(`:
//     image generateContent, video interactions, and the video Files API
//     poll/download all target GEMINI_HOST. No other host is ever contacted.
//   * Writes ONLY under the CALLER's own workspace output/genai-images and
//     output/genai-videos (server-derived from a validated `agent_name` param).
//     The caller never supplies a path — only its name, sanitized to ^[a-z0-9_-]+$.
//   * API key read from env GEMINI_API_KEY, never a param, never logged, never returned.
//   * n capped at 4; aspect_ratio allowlisted; reference_image size-capped.
//   * generate_video is a PAID call gated behind a required confirmed:true arg
//     (deliberation speed-bump; low API balance is the real cost backstop).
//   * Errors are typed strings, never silent empty files.
//
// Zero deps: Node >=18 globals only (fetch, Buffer, fs, path, process).

import { readFileSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { resolve, join, extname } from "node:path";

// ---- constants ------------------------------------------------------------
const GEMINI_HOST = "https://generativelanguage.googleapis.com";
// Image models the caller may pick. Keys are the accepted `model` values; the
// full model ID is also accepted. Anything else is a hard error — never a
// silent fallback to the default. IDs doc-verified 2026-09-23 (ai.google.dev).
// Prices are approximate per-image at 1K, for the schema description only.
const IMAGE_MODELS = {
  pro: { id: "gemini-3-pro-image-preview", price: "~13¢" },
  flash: { id: "gemini-3.1-flash-image", price: "~7¢" },
  "flash-lite": { id: "gemini-3.1-flash-lite-image", price: "~3.4¢" },
};
// Default stays Pro: marketing art depends on it and must not change silently.
const DEFAULT_IMAGE_MODEL = "pro";
function resolveImageModel(arg) {
  if (arg === undefined || arg === null || arg === "") return IMAGE_MODELS[DEFAULT_IMAGE_MODEL].id;
  if (typeof arg === "string") {
    if (Object.hasOwn(IMAGE_MODELS, arg)) return IMAGE_MODELS[arg].id;
    const byId = Object.values(IMAGE_MODELS).find((m) => m.id === arg);
    if (byId) return byId.id;
  }
  throw new Error(
    `invalid model: ${JSON.stringify(arg)} (allowed: ${Object.keys(IMAGE_MODELS).join(", ")})`
  );
}
const MAX_N = 4;
const MAX_REF_BYTES = 20 * 1024 * 1024; // 20 MB
// Video (Gemini Omni Flash, Interactions API) — doc-pinned 2026-07-06.
const VIDEO_MODEL = "gemini-omni-flash-preview";
const ALLOWED_VIDEO_ASPECT = new Set(["16:9", "9:16"]);
const VIDEO_TASKS = new Set(["text_to_video", "image_to_video", "reference_to_video"]);
const VIDEO_POLL_INTERVAL_MS = 5000;
const VIDEO_POLL_MAX_TRIES = 20; // 20 * 5s = 100s, under the 120s MCP ceiling
const ALLOWED_ASPECT = new Set([
  "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
]);
const MIME_BY_EXT = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
};
const AGENT_NAME_RE = /^[a-z0-9_-]+$/;

// The daemon connects MCP servers ONCE, globally — this process is shared by
// every agent, spawned with env_clear() + a fixed env whitelist and NO per-agent
// cwd. So the server is identity-blind: it cannot infer which agent called it.
//
// Delivery model (option X): each image lands in the CALLER's OWN workspace
// output dir — `~/.openfang/workspaces/<agent_name>/output/genai-images/` —
// because that is exactly the one path the attach gate already trusts for that
// agent (allow-root = the caller's real, unspoofable workspace). So the calling
// agent can `channel_send`-attach straight from there with NO `cp` hop and NO
// core rebuild, mirroring the pdf-attach pattern.
//
// The caller supplies only its NAME (validated ^[a-z0-9_-]+$), never a path —
// the anti-exfil guardrail survives intact. A spoofed name can at worst write
// nuisance bytes into a sibling's output/, but CANNOT attach them (the liar's
// own attach allow-root is its real workspace). Absent name → _unattributed/.
//
// OPENFANG_OUTPUT_ROOT still overrides everything (single-tenant escape hatch).
const WORKSPACES_BASE = process.env.HOME
  ? join(process.env.HOME, ".openfang", "workspaces")
  : null;
const OUTPUT_ROOT_OVERRIDE = process.env.OPENFANG_OUTPUT_ROOT || null;
const API_KEY = process.env.GEMINI_API_KEY;

// ---- stdio JSON-RPC plumbing ----------------------------------------------
// MCP stdio: one JSON message per line on stdin; one JSON response per line on
// stdout. stderr is free for logging (never the key).
function send(msg) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}
function result(id, res) {
  send({ jsonrpc: "2.0", id, result: res });
}
function error(id, code, message) {
  send({ jsonrpc: "2.0", id, error: { code, message } });
}
function log(...args) {
  process.stderr.write("[genai-image] " + args.join(" ") + "\n");
}

// ---- the tool -------------------------------------------------------------
const TOOL = {
  name: "generate_image",
  description:
    "Generate one or more images from a text prompt (optionally editing a " +
    "local reference image) via Google's Gemini image model. Writes PNG files " +
    "under the CALLING agent's own output/genai-images directory and returns " +
    "their paths, so the agent can attach them directly (no copy step). Pass " +
    "your own agent name as `agent_name`. Does NOT post to any channel — the " +
    "calling agent handles delivery.",
  inputSchema: {
    type: "object",
    properties: {
      prompt: { type: "string", description: "Text description of the image." },
      agent_name: {
        type: "string",
        description:
          "Your own agent name (^[a-z0-9_-]+$). Determines the output dir: " +
          "~/.openfang/workspaces/<agent_name>/output/genai-images/. Omit and " +
          "images land in a shared _unattributed/ dir you may not be able to attach from.",
      },
      aspect_ratio: {
        type: "string",
        description: "One of 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9. Default 1:1.",
      },
      n: { type: "integer", description: "How many images (1-4). Default 1." },
      model: {
        type: "string",
        enum: Object.keys(IMAGE_MODELS),
        description:
          "Image model. 'pro' (gemini-3-pro-image-preview, ~13¢/image) is the " +
          "DEFAULT and the marketing-art model. 'flash' (gemini-3.1-flash-image, " +
          "~7¢) is the middle option. 'flash-lite' (gemini-3.1-flash-lite-image, " +
          "~3.4¢) is cheapest, 1K only. Cost is per image, so n multiplies it.",
      },
      filename_slug: {
        type: "string",
        description: "Optional slug for the output filename (sanitized server-side).",
      },
      reference_image: {
        type: "string",
        description: "Optional path to a local image to edit/vary (png/jpg/webp, <=20MB).",
      },
    },
    required: ["prompt"],
  },
};

const VIDEO_TOOL = {
  name: "generate_video",
  description:
    "Generate a short video (3-10s, 720p) from a text prompt, optionally " +
    "animating a local reference image (image-to-video), via Google's Gemini " +
    "Omni Flash model. Writes an .mp4 under the CALLING agent's own " +
    "output/genai-videos directory and returns its path, so the agent can " +
    "attach it directly (no copy step). Pass your own agent name as " +
    "`agent_name`. PAID CALL (~$0.80/clip): you MUST pass `confirmed: true` to " +
    "proceed — the tool refuses otherwise. Does NOT post to any channel — the " +
    "calling agent handles delivery.",
  inputSchema: {
    type: "object",
    properties: {
      prompt: { type: "string", description: "Text description / motion of the video." },
      confirmed: {
        type: "boolean",
        description:
          "REQUIRED gate. Must be true. This is a paid generation (~$0.80/clip); " +
          "passing true is your explicit go-ahead. Any other value refuses the call.",
      },
      agent_name: {
        type: "string",
        description:
          "Your own agent name (^[a-z0-9_-]+$). Determines the output dir: " +
          "~/.openfang/workspaces/<agent_name>/output/genai-videos/. Omit and " +
          "videos land in a shared _unattributed/ dir you may not be able to attach from.",
      },
      aspect_ratio: {
        type: "string",
        description: "One of 16:9, 9:16. Default 16:9.",
      },
      reference_image: {
        type: "string",
        description:
          "Optional path to a local image to animate (png/jpg/webp, <=20MB). " +
          "Triggers image-to-video.",
      },
      task: {
        type: "string",
        description:
          "Optional generation task: text_to_video, image_to_video, " +
          "reference_to_video. Defaults to image_to_video when a reference_image " +
          "is given, else text_to_video.",
      },
      delivery: {
        type: "string",
        description:
          "How the result is returned: 'inline' (default, best for clips <=4MB) " +
          "or 'uri' (Google-hosted; poll-until-ACTIVE then download — use for " +
          "clips >4MB).",
      },
      filename_slug: {
        type: "string",
        description: "Optional slug for the output filename (sanitized server-side).",
      },
    },
    required: ["prompt", "confirmed"],
  },
};

function sanitizeSlug(slug) {
  if (!slug || typeof slug !== "string") return "";
  return slug
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
}

// Resolve the server-derived output root for this call.
//   1. OPENFANG_OUTPUT_ROOT wins if set (single-tenant escape hatch).
//   2. Otherwise: caller's own workspace output, keyed by validated agent_name.
//      - absent/empty  -> _unattributed/  (safe shared fallback)
//      - present+valid -> <agent_name>/   (matches ^[a-z0-9_-]+$; no traversal)
//      - present+bad   -> hard error (surfaces the mistake to the caller)
function resolveOutputRoot(agentNameArg) {
  if (OUTPUT_ROOT_OVERRIDE) return OUTPUT_ROOT_OVERRIDE;
  if (!WORKSPACES_BASE)
    throw new Error("no output root: set OPENFANG_OUTPUT_ROOT or HOME in environment");

  let name;
  if (agentNameArg === undefined || agentNameArg === null || agentNameArg === "") {
    name = "_unattributed";
  } else if (typeof agentNameArg === "string" && AGENT_NAME_RE.test(agentNameArg)) {
    name = agentNameArg;
  } else {
    throw new Error(
      `invalid agent_name: must match ^[a-z0-9_-]+$ (got ${JSON.stringify(agentNameArg)})`
    );
  }
  return join(WORKSPACES_BASE, name, "output");
}

function today() {
  const d = new Date();
  const p = (x) => String(x).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function loadReference(refPath) {
  const abs = resolve(refPath);
  if (!existsSync(abs)) throw new Error(`reference_image not found: ${refPath}`);
  const ext = extname(abs).toLowerCase();
  const mime = MIME_BY_EXT[ext];
  if (!mime) throw new Error(`unsupported reference_image type: ${ext} (png/jpg/webp only)`);
  const buf = readFileSync(abs);
  if (buf.length > MAX_REF_BYTES)
    throw new Error(`reference_image too large: ${buf.length} bytes (max ${MAX_REF_BYTES})`);
  return { mime, data: buf.toString("base64") };
}

async function callGemini(model, prompt, aspect, ref) {
  const parts = [];
  if (ref) parts.push({ inline_data: { mime_type: ref.mime, data: ref.data } });
  parts.push({ text: prompt });
  const body = {
    contents: [{ parts }],
    generationConfig: {
      responseModalities: ["IMAGE"],
      imageConfig: { aspectRatio: aspect },
    },
  };
  const url = `${GEMINI_HOST}/v1beta/models/${model}:generateContent`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-goog-api-key": API_KEY, // key in header, not the URL — stays out of logs
    },
    body: JSON.stringify(body),
  });
  const text = await resp.text();
  if (!resp.ok) {
    let detail = text;
    try {
      detail = JSON.parse(text)?.error?.message || text;
    } catch {}
    throw new Error(`gemini ${resp.status}: ${detail.slice(0, 400)}`);
  }
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error("gemini returned non-JSON body");
  }
  // Surface safety/quota blocks explicitly instead of writing an empty file.
  const cand = json?.candidates?.[0];
  if (!cand) {
    const reason = json?.promptFeedback?.blockReason || "no candidate returned";
    throw new Error(`gemini produced no image: ${reason}`);
  }
  const imgPart = (cand.content?.parts || []).find((p) => p.inlineData || p.inline_data);
  const inline = imgPart?.inlineData || imgPart?.inline_data;
  if (!inline?.data) {
    const fr = cand.finishReason || "unknown";
    throw new Error(`gemini candidate had no image data (finishReason=${fr})`);
  }
  return Buffer.from(inline.data, "base64");
}

async function generateImage(args) {
  if (!API_KEY) throw new Error("GEMINI_API_KEY not set in environment");

  const prompt = args?.prompt;
  if (!prompt || typeof prompt !== "string" || !prompt.trim())
    throw new Error("prompt is required");

  const outputRoot = resolveOutputRoot(args?.agent_name);

  const model = resolveImageModel(args?.model);

  const aspect = args?.aspect_ratio || "1:1";
  if (!ALLOWED_ASPECT.has(aspect))
    throw new Error(`invalid aspect_ratio: ${aspect}`);

  let n = Number.isInteger(args?.n) ? args.n : 1;
  if (n < 1) n = 1;
  if (n > MAX_N) n = MAX_N;

  const ref = args?.reference_image ? loadReference(args.reference_image) : null;

  const slug = sanitizeSlug(args?.filename_slug) || "image";
  const dir = join(outputRoot, "genai-images");
  mkdirSync(dir, { recursive: true });

  const date = today();
  const paths = [];
  for (let i = 0; i < n; i++) {
    const bytes = await callGemini(model, prompt, aspect, ref);
    const suffix = n > 1 ? `-${i + 1}` : "";
    const name = `${date}-${slug}${suffix}.png`;
    const outPath = join(dir, name);
    writeFileSync(outPath, bytes);
    paths.push({ path: outPath, bytes: bytes.length });
    log(`wrote ${outPath} (${bytes.length} bytes)`);
  }

  return {
    model,
    prompt,
    aspect_ratio: aspect,
    n: paths.length,
    images: paths,
  };
}

// ---- video (Gemini Omni Flash, Interactions API) --------------------------
// Pull the single video content item out of an Interactions response's steps.
function extractVideoContent(json) {
  const steps = Array.isArray(json?.steps) ? json.steps : [];
  for (const step of steps) {
    if (step?.type !== "model_output") continue;
    const content = Array.isArray(step.content) ? step.content : [];
    const vid = content.find((c) => c?.type === "video");
    if (vid) return vid;
  }
  return null;
}

// Poll the Files API until the generated file is ACTIVE, then download bytes.
// Each GET returns in seconds; capped at VIDEO_POLL_MAX_TRIES to stay under the
// 120s MCP tool ceiling. `downloadUri` is the full ...:download?alt=media URL
// carried in the CREATION response (re-fetching the interaction won't return it).
async function pollAndDownloadVideo(downloadUri) {
  const m = /\/files\/([^:/?]+)/.exec(downloadUri);
  if (!m) throw new Error(`could not parse file id from uri: ${downloadUri.slice(0, 120)}`);
  const fileId = m[1];
  const statusUrl = `${GEMINI_HOST}/v1beta/files/${fileId}`;
  for (let tries = 0; tries < VIDEO_POLL_MAX_TRIES; tries++) {
    const resp = await fetch(statusUrl, { headers: { "x-goog-api-key": API_KEY } });
    const text = await resp.text();
    if (!resp.ok) throw new Error(`files.get ${resp.status}: ${text.slice(0, 300)}`);
    let state;
    try {
      state = JSON.parse(text)?.state;
    } catch {
      throw new Error("files.get returned non-JSON body");
    }
    if (state === "ACTIVE") break;
    if (state === "FAILED") throw new Error("video generation failed (file state FAILED)");
    if (tries === VIDEO_POLL_MAX_TRIES - 1)
      throw new Error(
        `video still ${state} after ${VIDEO_POLL_MAX_TRIES} polls (~${
          (VIDEO_POLL_MAX_TRIES * VIDEO_POLL_INTERVAL_MS) / 1000
        }s); retry or raise the MCP timeout for this call`
      );
    await new Promise((r) => setTimeout(r, VIDEO_POLL_INTERVAL_MS));
  }
  // downloadUri already carries :download?alt=media — fetch it with the key.
  const dl = await fetch(downloadUri, { headers: { "x-goog-api-key": API_KEY } });
  if (!dl.ok) throw new Error(`file download ${dl.status}`);
  return Buffer.from(await dl.arrayBuffer());
}

async function callGeminiVideo(prompt, aspect, ref, task, delivery) {
  // Interactions `input`: an array when animating an image, else a bare string.
  let input;
  if (ref) {
    input = [
      { type: "image", data: ref.data, mime_type: ref.mime },
      { type: "text", text: prompt },
    ];
  } else {
    input = prompt;
  }
  const responseFormat = { type: "video", aspect_ratio: aspect };
  if (delivery === "uri") responseFormat.delivery = "uri";

  const body = {
    model: VIDEO_MODEL,
    input,
    // synchronous unary generation — fastest path, keeps no editable state.
    // store MUST be true for URI delivery: the clip is persisted to the Files
    // API so pollAndDownloadVideo can fetch it. Inline delivery keeps store:false
    // (no persisted state — bytes come back in the response body).
    background: false,
    store: delivery === "uri",
    stream: false,
    response_format: responseFormat,
  };
  if (ref || task) body.generation_config = { video_config: { task } };

  const url = `${GEMINI_HOST}/v1beta/interactions`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-goog-api-key": API_KEY, // key in header, not the URL — stays out of logs
    },
    body: JSON.stringify(body),
  });
  const text = await resp.text();
  if (!resp.ok) {
    let detail = text;
    try {
      detail = JSON.parse(text)?.error?.message || text;
    } catch {}
    throw new Error(`omni ${resp.status}: ${detail.slice(0, 400)}`);
  }
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error("omni returned non-JSON body");
  }
  const vid = extractVideoContent(json);
  if (!vid) {
    const status = json?.status || "unknown";
    throw new Error(`omni produced no video (status=${status})`);
  }
  if (delivery === "uri") {
    if (!vid.uri) throw new Error("omni delivery:uri returned no uri");
    return await pollAndDownloadVideo(vid.uri);
  }
  if (!vid.data) throw new Error("omni inline delivery returned no data");
  return Buffer.from(vid.data, "base64");
}

async function generateVideo(args) {
  if (!API_KEY) throw new Error("GEMINI_API_KEY not set in environment");

  // Confirm-gate: paid call. The server is identity-blind, so this is a
  // deliberation speed-bump, not a human-in-loop — the real cost backstop is a
  // low API balance. Still forces the caller to opt in explicitly, per call.
  if (args?.confirmed !== true)
    throw new Error(
      "generate_video is a PAID call (~$0.80/clip) and requires confirmed:true — " +
        "re-invoke with confirmed:true to proceed"
    );

  const prompt = args?.prompt;
  if (!prompt || typeof prompt !== "string" || !prompt.trim())
    throw new Error("prompt is required");

  const outputRoot = resolveOutputRoot(args?.agent_name);

  const aspect = args?.aspect_ratio || "16:9";
  if (!ALLOWED_VIDEO_ASPECT.has(aspect))
    throw new Error(`invalid aspect_ratio: ${aspect} (16:9 or 9:16 only)`);

  const ref = args?.reference_image ? loadReference(args.reference_image) : null;

  let task = args?.task;
  if (task && !VIDEO_TASKS.has(task))
    throw new Error(`invalid task: ${task} (text_to_video|image_to_video|reference_to_video)`);
  if (!task) task = ref ? "image_to_video" : "text_to_video";

  const delivery = args?.delivery === "uri" ? "uri" : "inline";

  const slug = sanitizeSlug(args?.filename_slug) || "video";
  const dir = join(outputRoot, "genai-videos");
  mkdirSync(dir, { recursive: true });

  const bytes = await callGeminiVideo(prompt, aspect, ref, task, delivery);
  const name = `${today()}-${slug}.mp4`;
  const outPath = join(dir, name);
  writeFileSync(outPath, bytes);
  log(`wrote ${outPath} (${bytes.length} bytes)`);

  return {
    model: VIDEO_MODEL,
    prompt,
    aspect_ratio: aspect,
    task,
    delivery,
    video: { path: outPath, bytes: bytes.length },
  };
}

// ---- dispatch -------------------------------------------------------------
async function handle(msg) {
  const { id, method, params } = msg;

  if (method === "initialize") {
    result(id, {
      protocolVersion: params?.protocolVersion || "2024-11-05",
      capabilities: { tools: {} },
      serverInfo: { name: "genai-image", version: "1.3.0" },
    });
    return;
  }
  if (method === "notifications/initialized" || method?.startsWith("notifications/")) {
    return; // notifications get no response
  }
  if (method === "ping") {
    result(id, {});
    return;
  }
  if (method === "tools/list") {
    result(id, { tools: [TOOL, VIDEO_TOOL] });
    return;
  }
  if (method === "tools/call") {
    const name = params?.name;
    const handlers = { generate_image: generateImage, generate_video: generateVideo };
    const fn = handlers[name];
    if (!fn) {
      error(id, -32602, `unknown tool: ${name}`);
      return;
    }
    try {
      const res = await fn(params?.arguments || {});
      result(id, {
        content: [{ type: "text", text: JSON.stringify(res, null, 2) }],
      });
    } catch (e) {
      // Typed tool error — visible to the caller, not a protocol crash.
      result(id, {
        isError: true,
        content: [{ type: "text", text: `${name} failed: ${e.message}` }],
      });
    }
    return;
  }
  if (id !== undefined && id !== null) {
    error(id, -32601, `method not found: ${method}`);
  }
}

// ---- stdin line reader ----------------------------------------------------
let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let nl;
  while ((nl = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (!line) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      log("skipped non-JSON line");
      continue;
    }
    handle(msg).catch((e) => log("handler error:", e.message));
  }
});
process.stdin.on("end", () => process.exit(0));
log(
  `ready (image default=${IMAGE_MODELS[DEFAULT_IMAGE_MODEL].id}, video=${VIDEO_MODEL}, mode=${
    OUTPUT_ROOT_OVERRIDE ? "override:" + OUTPUT_ROOT_OVERRIDE : "per-caller-workspace"
  })`
);
