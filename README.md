# NVIDIA NIM OpenAI Proxy

A lightweight Node.js proxy that makes NVIDIA NIM easier to use from apps that expect an OpenAI-style `/v1/chat/completions` endpoint.

This version is tuned for **Janitor AI** and **SillyTavern** by converting NVIDIA `reasoning_content` into normal assistant `content` wrapped in `<think>...</think>` tags.

## What it supports

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- streaming pass-through
- reasoning display as `<think>` blocks for clients that only read `content`
- simple model aliases for Janitor AI / SillyTavern style clients

## Default behavior

By default this proxy now:

- maps `gpt-4o` to `qwen/qwen3-next-80b-a3b-thinking`
- enables `extra_body.chat_template_kwargs.enable_thinking=true` for models other than GLM 5.3
- rewrites `reasoning_content` into visible `<think>` text

If a model does not like the `enable_thinking` flag, set `ENABLE_THINKING_MODE=false`.
This only stops automatically adding the flag; it does not force reasoning off.

For `z-ai/glm-5.3`, the proxy defaults to `clear_thinking=true` and
`reasoning_effort=low`, with reasoning displayed because `SHOW_REASONING=true`.
GLM 5.3 always reasons, so the proxy omits its unsupported `enable_thinking` flag.
Explicit request values override the GLM 5.3 environment defaults. Both top-level
parameters and SDK-style `extra_body` parameters are accepted for this model;
the proxy flattens them into the raw JSON NVIDIA expects, with top-level values
taking precedence. Other models retain their existing defaults.

## Quick start

```bash
npm install
npm start
```

Server runs on `http://localhost:3000` by default.

## Required env vars

- `NIM_API_KEY` - your NVIDIA API key

## Optional env vars

- `PORT=3000`
- `NIM_API_BASE=https://integrate.api.nvidia.com/v1`
- `SHOW_REASONING=true`
- `CLEAR_THINKING=true` - GLM 5.3 default; clears reasoning from earlier turns, preserving answers
- `REASONING_EFFORT=low` - GLM 5.3 default; accepts `low`, `high`, or `max`
- `ENABLE_THINKING_MODE=true`
- `INCLUDE_RAW_NIM_MODELS=true`
- `REQUEST_TIMEOUT_MS=120000`
- `DEBUG=false`

## Example curl

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Think through this briefly, then answer: how many r in strawberry?"}
    ]
  }'
```

## Janitor / SillyTavern notes

- Use your Railway URL plus `/v1` as the base URL.
- Start with model `gpt-4o` if you want visible thinking.
- If you want cleaner answers without visible thinking, set `SHOW_REASONING=false` and redeploy.
- If a specific model fails because it does not support the reasoning toggle, set `ENABLE_THINKING_MODE=false` and redeploy.

## Vercel deploy notes

- Select `z-ai/glm-5.3` in your client to use these model-specific defaults.
- In Project Settings → Environment Variables, set `CLEAR_THINKING=true`,
  `REASONING_EFFORT=low`, and `SHOW_REASONING=true`, alongside `NIM_API_KEY`.
- Local `.env` files are ignored by Git and do not sync to Vercel.
- Push to the connected production branch to trigger deployment when automatic
  Git deployments are enabled. Redeploy after changing environment variables.
- `/health` shows `glm_5_3_defaults` and `show_reasoning` for verification.

## Railway deploy notes

- New Project
- Deploy from GitHub repo
- Add `NIM_API_KEY` in Variables
- Railway should auto-detect Node
- Start command: `npm start`
- Leave **Serverless disabled** if you want it always on

## Render deploy notes

- New Web Service
- Connect repo
- Build command: `npm install`
- Start command: `npm start`
- Add `NIM_API_KEY` as an env var
- Free tier sleeps on idle, so do not use free for always-on chat proxy
