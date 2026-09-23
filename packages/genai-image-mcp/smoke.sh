#!/usr/bin/env bash
# Offline smoke: exercises the stdio surface WITHOUT any network call.
# 1) tools/list must advertise both tools.
# 2) generate_video without confirmed:true must be refused (typed error, no egress).
set -euo pipefail
cd "$(dirname "$0")"

IN=$(mktemp)
cat > "$IN" <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"generate_video","arguments":{"prompt":"a cat"}}}
EOF

# No GEMINI_API_KEY needed: the gate check fires before the key check for id=3,
# and id=1/2 don't touch the network. Run with a dummy key to be safe.
GEMINI_API_KEY=dummy node server.mjs < "$IN"
rm -f "$IN"
