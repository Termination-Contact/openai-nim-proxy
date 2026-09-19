const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function load(env = {}) {
  const routes = {};
  const app = { use() {}, get() {}, post(p, handler) { routes[p] = handler; }, listen() {} };
  const express = () => app;
  express.json = () => () => {};
  let sent;
  const axios = async (options) => {
    sent = JSON.parse(JSON.stringify(options.data));
    return { status: 200, data: { choices: [{ message: {
      content: 'Answer', reasoning_content: 'Reasoning'
    } }] } };
  };
  const context = vm.createContext({
    require(name) {
      if (name === 'fs') return { existsSync: () => false };
      if (name === 'express') return express;
      if (name === 'cors') return () => () => {};
      if (name === 'axios') return axios;
      return require(name);
    },
    __dirname: path.join(__dirname, '..'),
    process: { env: { NIM_API_KEY: 'test', ...env } },
    console: { log() {}, warn() {}, error() {} }
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../server.js'), 'utf8'), context);
  return {
    context,
    async request(body) {
      let result;
      await routes['/v1/chat/completions']({ body: {
        model: 'z-ai/glm-5.3', messages: [{ role: 'user', content: 'Hello' }], ...body
      } }, { status() { return this; }, json(value) { result = value; } });
      return { sent, result };
    }
  };
}

test('GLM 5.3 sends raw defaults and displays reasoning', async () => {
  const { sent, result } = await load().request({});
  assert.deepEqual(sent.chat_template_kwargs, { clear_thinking: true });
  assert.equal(sent.reasoning_effort, 'low');
  assert.equal(sent.extra_body, undefined);
  assert.equal(sent.model, 'z-ai/glm-5.3');
  assert.equal(result.choices[0].message.content, '<think>\nReasoning\n</think>\n\nAnswer');
});

test('environment settings work, including capitalized False', async () => {
  const { sent } = await load({ CLEAR_THINKING: 'False', REASONING_EFFORT: 'high' }).request({});
  assert.equal(sent.chat_template_kwargs.clear_thinking, false);
  assert.equal(sent.reasoning_effort, 'high');
  assert.throws(() => load({ REASONING_EFFORT: 'none' }), /must be low, high, or max/);
});

test('request overrides survive flattening; top-level wins and model cannot change', async () => {
  const { sent } = await load().request({
    extra_body: { model: 'wrong', reasoning_effort: 'high', chat_template_kwargs: {
      clear_thinking: true, enable_thinking: false
    } },
    reasoning_effort: 'max', chat_template_kwargs: { clear_thinking: false }
  });
  assert.equal(sent.model, 'z-ai/glm-5.3');
  assert.equal(sent.reasoning_effort, 'max');
  assert.deepEqual(sent.chat_template_kwargs, { clear_thinking: false });
  const nested = await load().request({ extra_body: {
    reasoning_effort: 'high', chat_template_kwargs: { clear_thinking: false }
  } });
  assert.equal(nested.sent.reasoning_effort, 'high');
  assert.equal(nested.sent.chat_template_kwargs.clear_thinking, false);
});

test('other models retain their previous defaults', async () => {
  const { sent } = await load().request({ model: 'gpt-4o' });
  assert.equal(sent.model, 'qwen/qwen3-next-80b-a3b-thinking');
  assert.deepEqual(sent.extra_body, { chat_template_kwargs: { enable_thinking: true } });
  assert.equal(sent.reasoning_effort, undefined);
  assert.equal(sent.chat_template_kwargs, undefined);
});

test('streamed reasoning stays visible', () => {
  const { context } = load();
  const output = vm.runInContext(`(() => {
    const state = createStreamState();
    return reasoningChunkToContent({reasoning_content:'Reasoning'}, state, 0).content
      + reasoningChunkToContent({content:'Answer'}, state, 0).content;
  })()`, context);
  assert.equal(output, '<think>\nReasoning\n</think>\n\nAnswer');
});
