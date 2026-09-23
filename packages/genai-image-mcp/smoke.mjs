// Offline smoke harness — no network. Spawns server.mjs, drives its stdio,
// asserts: (1) tools/list advertises both tools, (2) generate_video without
// confirmed:true is refused with a typed error before any egress,
// (3) generate_image advertises the model allowlist, (4) an unknown or
// non-string model is refused before any egress — never a silent fallback.
// Run: node smoke.mjs   (dummy key so the gate — not the key check — is what fires)
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const child = spawn("node", [join(here, "server.mjs")], {
  env: { ...process.env, GEMINI_API_KEY: "dummy" },
  stdio: ["pipe", "pipe", "inherit"],
});

const imageCall = (id, model) => ({
  jsonrpc: "2.0",
  id,
  method: "tools/call",
  params: { name: "generate_image", arguments: { prompt: "a cat", agent_name: "smoke-test", model } },
});

const reqs = [
  { jsonrpc: "2.0", id: 1, method: "initialize", params: {} },
  { jsonrpc: "2.0", id: 2, method: "tools/list" },
  {
    jsonrpc: "2.0",
    id: 3,
    method: "tools/call",
    params: { name: "generate_video", arguments: { prompt: "a cat" } },
  },
  imageCall(4, "ultra"),
  imageCall(5, 42),
];
for (const r of reqs) child.stdin.write(JSON.stringify(r) + "\n");
child.stdin.end();

let out = "";
child.stdout.setEncoding("utf8");
child.stdout.on("data", (c) => (out += c));
child.stdout.on("end", () => {
  const msgs = out.trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));
  const byId = (id) => msgs.find((m) => m.id === id);
  const list = byId(2);
  const call = byId(3);
  const tools = list?.result?.tools || [];
  const names = tools.map((t) => t.name).sort();

  const checks = [];
  checks.push(["tools/list has both tools", JSON.stringify(names) === JSON.stringify(["generate_image", "generate_video"])]);
  const videoTool = tools.find((t) => t.name === "generate_video");
  checks.push(["generate_video requires prompt+confirmed", JSON.stringify(videoTool?.inputSchema?.required?.slice().sort()) === JSON.stringify(["confirmed", "prompt"])]);
  const gateText = call?.result?.content?.[0]?.text || "";
  checks.push(["confirm-gate refuses w/o confirmed", call?.result?.isError === true && /confirmed:true/.test(gateText)]);

  const imageTool = tools.find((t) => t.name === "generate_image");
  const modelEnum = imageTool?.inputSchema?.properties?.model?.enum || [];
  checks.push(["generate_image advertises model allowlist", JSON.stringify(modelEnum.slice().sort()) === JSON.stringify(["flash", "flash-lite", "pro"])]);
  for (const [id, label] of [[4, '"ultra"'], [5, "42"]]) {
    const r = byId(id);
    const text = r?.result?.content?.[0]?.text || r?.error?.message || "";
    const refused = (r?.result?.isError === true || r?.error !== undefined) && /invalid model/.test(text);
    checks.push([`model ${label} refused before egress`, refused]);
  }

  let ok = true;
  for (const [label, pass] of checks) {
    console.log(`${pass ? "PASS" : "FAIL"}  ${label}`);
    if (!pass) ok = false;
  }
  console.log(ok ? "\nALL PASS" : "\nFAILURES PRESENT");
  process.exit(ok ? 0 : 1);
});
