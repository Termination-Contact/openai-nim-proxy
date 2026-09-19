const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');
const axios = require('axios');

function loadEnvFile(filePath) {
  if (!fs.existsSync(filePath)) return;

  const content = fs.readFileSync(filePath, 'utf8');
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const eqIndex = line.indexOf('=');
    if (eqIndex === -1) continue;

    const key = line.slice(0, eqIndex).trim();
    if (!key) continue;

    let value = line.slice(eqIndex + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }

    if (process.env[key] === undefined) {
      process.env[key] = value;
    }
  }
}

loadEnvFile(path.join(__dirname, '.env'));

const app = express();
const PORT = process.env.PORT || 3000;
const NIM_API_BASE = (process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1').replace(/\/$/, '');
const NIM_API_KEY = process.env.NIM_API_KEY;
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS || 120000);
const DEBUG = String(process.env.DEBUG || 'false').toLowerCase() === 'true';

// Janitor / SillyTavern friendly toggles
const SHOW_REASONING = String(process.env.SHOW_REASONING || 'true').toLowerCase() === 'true';
const ENABLE_THINKING_MODE = String(process.env.ENABLE_THINKING_MODE || 'true').toLowerCase() === 'true';
const CLEAR_THINKING = String(process.env.CLEAR_THINKING || 'true').toLowerCase() === 'true';
const REASONING_EFFORT = String(process.env.REASONING_EFFORT || 'low').trim().toLowerCase();
if (!['low', 'high', 'max'].includes(REASONING_EFFORT)) {
  throw new Error('REASONING_EFFORT must be low, high, or max');
}
const INCLUDE_RAW_NIM_MODELS = String(process.env.INCLUDE_RAW_NIM_MODELS || 'true').toLowerCase() === 'true';

if (!NIM_API_KEY) {
  console.warn('WARNING: NIM_API_KEY is not set. Requests to NVIDIA NIM will fail until you add it.');
}

app.use(cors());
app.use(express.json({ limit: '4mb' }));

// Use at least one alias that points to a reasoning-capable model so Janitor/ST can show <think> blocks.
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4o': 'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek-chat': 'deepseek-ai/deepseek-v3.2',
  'deepseek-reasoner': 'qwen/qwen3-next-80b-a3b-thinking',
  'qwen-thinking': 'qwen/qwen3-next-80b-a3b-thinking',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'openai/gpt-oss-20b': 'openai/gpt-oss-20b',
  'openai/gpt-oss-120b': 'openai/gpt-oss-120b',
  'moonshotai/kimi-k2-instruct-0905': 'moonshotai/kimi-k2-instruct-0905',
  'deepseek-ai/deepseek-v3.1': 'deepseek-ai/deepseek-v3.1',
  'qwen/qwen3-coder-480b-a35b-instruct': 'qwen/qwen3-coder-480b-a35b-instruct',
  'qwen/qwen3-next-80b-a3b-thinking': 'qwen/qwen3-next-80b-a3b-thinking',
  'z-ai/glm4.7': 'z-ai/glm4.7',
  'z-ai/glm5': 'z-ai/glm5',
  'z-ai/glm-5.3': 'z-ai/glm-5.3',
  'stepfun-ai/step-3.5-flash': 'stepfun-ai/step-3.5-flash',
  'minimaxai/minimax-m2.5': 'minimaxai/minimax-m2.5',
};

function resolveModel(requestedModel) {
  if (!requestedModel) return MODEL_MAPPING['gpt-4o'];
  return MODEL_MAPPING[requestedModel] || requestedModel;
}

function rewriteResponseModel(data, requestedModel) {
  if (!data || !requestedModel) return data;
  if (typeof data === 'object' && data.model) {
    data.model = requestedModel;
  }
  return data;
}

function buildError(status, message, extra = {}) {
  return {
    error: {
      message,
      type: extra.type || 'invalid_request_error',
      code: extra.code || status,
      ...extra
    }
  };
}

function clone(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function ensureReasoningSettings(body, upstreamModel) {
  const out = clone(body) || {};
  const extraBody = clone(out.extra_body) || {};

  if (upstreamModel === 'z-ai/glm-5.3') {
    // Axios sends raw JSON: SDK-style extra_body fields must be flattened.
    const merged = { ...extraBody, ...out };
    delete merged.extra_body;
    merged.chat_template_kwargs = {
      ...extraBody.chat_template_kwargs,
      ...out.chat_template_kwargs
    };
    // GLM 5.3 always reasons and does not use enable_thinking.
    delete merged.chat_template_kwargs.enable_thinking;
    if (merged.chat_template_kwargs.clear_thinking === undefined) {
      merged.chat_template_kwargs.clear_thinking = CLEAR_THINKING;
    }
    if (merged.reasoning_effort === undefined) {
      merged.reasoning_effort = REASONING_EFFORT;
    }
    // Extra parameters cannot change the model selected by the route.
    delete merged.model;
    return merged;
  }

  const chatTemplateKwargs = clone(extraBody.chat_template_kwargs) || {};

  if (ENABLE_THINKING_MODE && chatTemplateKwargs.enable_thinking === undefined) {
    chatTemplateKwargs.enable_thinking = true;
  }

  if (Object.keys(chatTemplateKwargs).length > 0) {
    extraBody.chat_template_kwargs = chatTemplateKwargs;
  }

  if (Object.keys(extraBody).length > 0) {
    out.extra_body = extraBody;
  }

  return out;
}

function pickChatBody(body, upstreamModel) {
  const allowed = [
    'messages',
    'temperature',
    'max_tokens',
    'max_completion_tokens',
    'stream',
    'top_p',
    'top_k',
    'min_p',
    'stop',
    'presence_penalty',
    'frequency_penalty',
    'seed',
    'n',
    'tools',
    'tool_choice',
    'response_format',
    'user',
    'extra_body',
    'chat_template_kwargs',
    'reasoning_effort',
    'include_reasoning',
    'logprobs'
  ];

  const out = {};
  for (const key of allowed) {
    if (body[key] !== undefined) out[key] = clone(body[key]);
  }
  return ensureReasoningSettings(out, upstreamModel);
}

function mergeReasoningIntoContent(message) {
  if (!message || typeof message !== 'object') return message;

  const cloned = { ...message };
  const content = typeof cloned.content === 'string' ? cloned.content : '';
  const reasoning = typeof cloned.reasoning_content === 'string' ? cloned.reasoning_content : '';

  if (SHOW_REASONING && reasoning) {
    cloned.content = `<think>\n${reasoning}\n</think>${content ? `\n\n${content}` : ''}`;
  } else {
    cloned.content = content;
  }

  delete cloned.reasoning_content;
  return cloned;
}

function transformNonStreamingPayload(payload, requestedModel) {
  const base = rewriteResponseModel(clone(payload), requestedModel);
  if (!base || !Array.isArray(base.choices)) return base;

  base.choices = base.choices.map((choice) => {
    const nextChoice = { ...choice };
    if (nextChoice.message) {
      nextChoice.message = mergeReasoningIntoContent(nextChoice.message);
    }
    return nextChoice;
  });

  return base;
}

function createStreamState() {
  return {
    reasoningStarted: new Map(),
    reasoningClosed: new Map()
  };
}

function reasoningChunkToContent(delta, state, choiceIndex) {
  if (!delta || typeof delta !== 'object') return null;

  const reasoning = typeof delta.reasoning_content === 'string' ? delta.reasoning_content : '';
  const content = typeof delta.content === 'string' ? delta.content : '';

  if (!SHOW_REASONING) {
    if (reasoning || Object.prototype.hasOwnProperty.call(delta, 'reasoning_content')) {
      delete delta.reasoning_content;
      if (!Object.prototype.hasOwnProperty.call(delta, 'content')) {
        delta.content = '';
      }
    }
    return delta;
  }

  let combined = '';
  const reasoningStarted = state.reasoningStarted.get(choiceIndex) === true;
  const reasoningClosed = state.reasoningClosed.get(choiceIndex) === true;

  if (reasoning) {
    if (!reasoningStarted) {
      combined += '<think>\n';
      state.reasoningStarted.set(choiceIndex, true);
      state.reasoningClosed.set(choiceIndex, false);
    }
    combined += reasoning;
  }

  if (content) {
    if (state.reasoningStarted.get(choiceIndex) === true && state.reasoningClosed.get(choiceIndex) !== true) {
      combined += '\n</think>\n\n';
      state.reasoningClosed.set(choiceIndex, true);
    }
    combined += content;
  }

  if (!combined && Object.prototype.hasOwnProperty.call(delta, 'reasoning_content')) {
    delete delta.reasoning_content;
    delta.content = '';
    return delta;
  }

  if (combined) {
    delta.content = combined;
  }

  if (Object.prototype.hasOwnProperty.call(delta, 'reasoning_content')) {
    delete delta.reasoning_content;
  }

  if (!content && delta.finish_reason && state.reasoningStarted.get(choiceIndex) === true && state.reasoningClosed.get(choiceIndex) !== true) {
    delta.content = `${delta.content || ''}\n</think>`;
    state.reasoningClosed.set(choiceIndex, true);
  }

  return delta;
}

function transformStreamingEvent(rawLine, state, requestedModel) {
  if (!rawLine.startsWith('data: ')) {
    return `${rawLine}\n`;
  }

  const payloadText = rawLine.slice(6).trim();
  if (payloadText === '[DONE]') {
    return 'data: [DONE]\n\n';
  }

  let payload;
  try {
    payload = JSON.parse(payloadText);
  } catch {
    return `${rawLine}\n`;
  }

  rewriteResponseModel(payload, requestedModel);

  if (Array.isArray(payload.choices)) {
    payload.choices = payload.choices.map((choice, index) => {
      const nextChoice = { ...choice };
      const choiceIndex = nextChoice.index ?? index;
      if (nextChoice.delta) {
        nextChoice.delta = reasoningChunkToContent({ ...nextChoice.delta }, state, choiceIndex);
      }
      return nextChoice;
    });
  }

  return `data: ${JSON.stringify(payload)}\n\n`;
}

async function fetchUpstreamModels() {
  if (!NIM_API_KEY || !INCLUDE_RAW_NIM_MODELS) return [];

  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      timeout: Math.min(REQUEST_TIMEOUT_MS, 15000),
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      validateStatus: () => true
    });

    if (response.status >= 200 && response.status < 300 && Array.isArray(response.data?.data)) {
      return response.data.data;
    }
  } catch (error) {
    if (DEBUG) {
      console.warn('Could not fetch upstream models:', error.message);
    }
  }

  return [];
}

app.get('/', (_req, res) => {
  res.json({
    name: 'NVIDIA NIM OpenAI Proxy',
    ok: true,
    show_reasoning: SHOW_REASONING,
    enable_thinking_mode: ENABLE_THINKING_MODE,
    endpoints: ['/health', '/v1/models', '/v1/chat/completions']
  });
});

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'nim-openai-proxy',
    upstream: NIM_API_BASE,
    has_api_key: Boolean(NIM_API_KEY),
    glm_5_3_defaults: { clear_thinking: CLEAR_THINKING, reasoning_effort: REASONING_EFFORT },
    show_reasoning: SHOW_REASONING,
    enable_thinking_mode: ENABLE_THINKING_MODE
  });
});

app.get('/v1/models', async (_req, res) => {
  const aliasModels = Object.keys(MODEL_MAPPING).map((alias) => ({
    id: alias,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nim-openai-proxy'
  }));

  const upstreamModels = await fetchUpstreamModels();
  const dedup = new Map();

  for (const model of [...aliasModels, ...upstreamModels]) {
    if (model?.id) dedup.set(model.id, model);
  }

  res.json({
    object: 'list',
    data: Array.from(dedup.values())
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  const requestedModel = req.body?.model || 'gpt-4o';
  const upstreamModel = resolveModel(requestedModel);
  const stream = Boolean(req.body?.stream);

  if (!Array.isArray(req.body?.messages)) {
    return res.status(400).json(buildError(400, 'messages must be an array'));
  }

  const upstreamBody = {
    model: upstreamModel,
    ...pickChatBody(req.body, upstreamModel)
  };

  if (DEBUG) {
    console.log('chat request', {
      requestedModel,
      upstreamModel,
      stream,
      messageCount: req.body.messages.length,
      show_reasoning: SHOW_REASONING,
      enable_thinking_mode: ENABLE_THINKING_MODE
    });
  }

  try {
    const response = await axios({
      method: 'post',
      url: `${NIM_API_BASE}/chat/completions`,
      data: upstreamBody,
      timeout: REQUEST_TIMEOUT_MS,
      responseType: stream ? 'stream' : 'json',
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      validateStatus: () => true
    });

    if (stream) {
      res.status(response.status);
      res.setHeader('Content-Type', response.headers['content-type'] || 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const state = createStreamState();
      let buffer = '';

      response.data.on('data', (chunk) => {
        buffer += chunk.toString('utf8');
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          res.write(transformStreamingEvent(line, state, requestedModel));
        }
      });

      response.data.on('end', () => {
        if (buffer.trim()) {
          res.write(transformStreamingEvent(buffer.trim(), state, requestedModel));
        }
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('stream error:', err.message);
        res.end();
      });
      return;
    }

    const payload = transformNonStreamingPayload(response.data, requestedModel);
    return res.status(response.status).json(payload);
  } catch (error) {
    console.error('proxy error:', error.message);

    const status = error.response?.status || 500;
    const upstreamMessage = error.response?.data?.error?.message || error.message || 'Internal server error';

    return res.status(status).json(buildError(status, upstreamMessage));
  }
});

app.use((req, res) => {
  res.status(404).json(buildError(404, `Endpoint ${req.method} ${req.path} not found`));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`NVIDIA NIM OpenAI proxy listening on port ${PORT}`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
