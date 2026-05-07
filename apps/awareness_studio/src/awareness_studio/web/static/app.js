/* Awareness Studio Control Panel — app.js */
'use strict';

const BASE = '';  // same origin

// ── State ────────────────────────────────────────────────────────────────────
let streaming = false;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const chatHistory   = document.getElementById('chat-history');
const questionInput = document.getElementById('question-input');
const sendBtn       = document.getElementById('send-btn');
const modeSelect    = document.getElementById('mode-select');
const kInput        = document.getElementById('k-input');
const streamToggle  = document.getElementById('stream-toggle');
const toolsToggle   = document.getElementById('tools-toggle');
const resultsPanel  = document.getElementById('results-panel');
const resultsHeader = document.getElementById('results-title');
const resultsBody   = document.getElementById('results-body');
const resultsClose  = document.getElementById('results-close');
const statusDot     = document.querySelector('.status-dot');
const statusLabel   = document.querySelector('.status-label');

// ── Helpers ───────────────────────────────────────────────────────────────────

function genId() {
  return 'req-' + Math.random().toString(36).slice(2, 10);
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderLabels(text) {
  return text
    .replace(/\[Direct teaching\]/g, '<span class="label-dt">[Direct teaching]</span>')
    .replace(/\[Method-synthesis\]/g, '<span class="label-ms">[Method-synthesis]</span>')
    .replace(/\[Hypothesis\]/g, '<span class="label-hy">[Hypothesis]</span>');
}

function showResults(title, content) {
  resultsHeader.textContent = title;
  resultsBody.textContent = typeof content === 'string'
    ? content : JSON.stringify(content, null, 2);
  resultsPanel.classList.add('open');
}

resultsClose.addEventListener('click', () => resultsPanel.classList.remove('open'));

// ── Health check ──────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const r = await fetch(BASE + '/health');
    const d = await r.json();
    statusDot.style.background = d.status === 'ok' ? 'var(--success)' : 'var(--error)';
    statusLabel.textContent = `backend=${d.backend} tools=${d.tools_enabled ? 'on' : 'off'}`;
  } catch {
    statusDot.style.background = 'var(--error)';
    statusLabel.textContent = 'offline';
  }
}

checkHealth();

// ── Chat message rendering ────────────────────────────────────────────────────

function appendUserMsg(text) {
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `<div class="msg-bubble">${escHtml(text)}</div>`;
  chatHistory.appendChild(el);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return el;
}

function appendAssistantMsg() {
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.innerHTML = `
    <div class="msg-bubble" id="current-bubble"></div>
    <div class="msg-meta" id="current-meta"></div>
    <div class="info-panels" id="current-panels"></div>
  `;
  chatHistory.appendChild(el);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return el;
}

function updateBubble(el, text) {
  const bubble = el.querySelector('.msg-bubble');
  bubble.innerHTML = renderLabels(escHtml(text));
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function finalizeMeta(el, meta) {
  const m = el.querySelector('.msg-meta');
  m.textContent = `mode=${meta.mode}  retrieved=${meta.retrieved ?? '?'}  id=${meta.request_id ?? '?'}`;
}

function renderSourcesPanel(el, sources) {
  if (!sources || sources.length === 0) return;
  const panels = el.querySelector('.info-panels');
  const p = document.createElement('div');
  p.className = 'panel';
  p.innerHTML = `
    <div class="panel-header" onclick="togglePanel(this)">
      Sources used <span>${sources.length}</span>
    </div>
    <div class="panel-body">
      ${sources.map(s => `<span class="source-chip">${escHtml(s)}</span>`).join('')}
    </div>`;
  panels.appendChild(p);
}

function renderToolCallsPanel(el, calls) {
  if (!calls || calls.length === 0) return;
  const panels = el.querySelector('.info-panels');
  const p = document.createElement('div');
  p.className = 'panel';
  const rows = calls.map(c => `
    <div class="tool-row">
      <span class="tool-name">${escHtml(c.tool_name)}</span>
      <span class="${c.success ? 'tool-ok' : 'tool-err'}">${c.success ? '✓' : '✗'}</span>
      <span style="color:var(--muted)">${escHtml(c.result_summary)}</span>
    </div>`).join('');
  p.innerHTML = `
    <div class="panel-header" onclick="togglePanel(this)">
      Tool calls <span>${calls.length}</span>
    </div>
    <div class="panel-body">${rows}</div>`;
  panels.appendChild(p);
}

window.togglePanel = function(header) {
  const body = header.nextElementSibling;
  body.classList.toggle('collapsed');
};

// Extract source chunk IDs from "## Sources used" section
function extractSources(text) {
  const m = text.match(/## Sources used\n([\s\S]*?)(?:\n##|$)/);
  if (!m) return [];
  return [...m[1].matchAll(/`([^`]+)`/g)].map(x => x[1]);
}

// ── SSE streaming chat ────────────────────────────────────────────────────────

function sendStreaming(question, mode, k, useTools) {
  const reqId = genId();
  appendUserMsg(question);
  const assistantEl = appendAssistantMsg();
  let buffer = '';
  let toolCalls = [];
  setBusy(true);

  fetch(BASE + '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, mode, k, stream: true, use_tools: useTools,
                           request_id: reqId }),
  }).then(resp => {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    function pump() {
      reader.read().then(({ done, value }) => {
        if (done) {
          renderSourcesPanel(assistantEl, extractSources(buffer));
          renderToolCallsPanel(assistantEl, toolCalls);
          finalizeMeta(assistantEl, { mode, request_id: reqId });
          setBusy(false);
          return;
        }
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') continue;
          try {
            const obj = JSON.parse(raw);
            if (obj.token !== undefined) {
              buffer += obj.token;
              updateBubble(assistantEl, buffer);
            }
            if (obj.tool_calls) {
              toolCalls = obj.tool_calls;
            }
          } catch { /* partial data */ }
        }
        pump();
      }).catch(err => {
        updateBubble(assistantEl, buffer + '\n\n[stream error: ' + err.message + ']');
        setBusy(false);
      });
    }
    pump();
  }).catch(err => {
    appendSystemMsg('Network error: ' + err.message);
    setBusy(false);
  });
}

// ── JSON (non-streaming) chat ─────────────────────────────────────────────────

async function sendJson(question, mode, k, useTools) {
  const reqId = genId();
  appendUserMsg(question);
  const assistantEl = appendAssistantMsg();
  setBusy(true);
  try {
    const r = await fetch(BASE + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, mode, k, stream: false, use_tools: useTools }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);

    updateBubble(assistantEl, d.answer);
    renderSourcesPanel(assistantEl, extractSources(d.answer));
    renderToolCallsPanel(assistantEl, d.tool_calls || []);
    finalizeMeta(assistantEl, {
      mode: d.mode,
      retrieved: d.retrieved,
      request_id: d.request_id || reqId,
    });
  } catch (err) {
    updateBubble(assistantEl, '[Error: ' + err.message + ']');
  } finally {
    setBusy(false);
  }
}

// ── Send dispatch ─────────────────────────────────────────────────────────────

function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || streaming) return;

  const mode     = modeSelect.value;
  const k        = parseInt(kInput.value, 10) || 8;
  const useTools = toolsToggle.checked;
  const doStream = streamToggle.checked;

  questionInput.value = '';
  questionInput.style.height = 'auto';

  if (doStream) {
    sendStreaming(question, mode, k, useTools);
  } else {
    sendJson(question, mode, k, useTools);
  }
}

sendBtn.addEventListener('click', sendMessage);
questionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = questionInput.scrollHeight + 'px';
});

function setBusy(busy) {
  streaming = busy;
  sendBtn.disabled = busy;
  sendBtn.textContent = busy ? '…' : 'Send';
}

// ── System messages ───────────────────────────────────────────────────────────

function appendSystemMsg(text) {
  const el = document.createElement('div');
  el.style.cssText = 'text-align:center;color:var(--muted);font-size:12px;padding:8px;';
  el.textContent = text;
  chatHistory.appendChild(el);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

// ── Sidebar commands ──────────────────────────────────────────────────────────

document.querySelectorAll('.sidebar-btn[data-cmd]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const cmd = btn.dataset.cmd;
    runCommand(cmd, btn);
  });
});

async function runCommand(cmd, btn) {
  const label = btn.querySelector('.label') || btn;
  const origText = label.textContent;
  label.textContent = '…';
  try {
    if (cmd === 'tools-list') {
      const d = await (await fetch(BASE + '/tools/list')).json();
      showResults('Tools List', d);

    } else if (cmd === 'lit-search') {
      const q = prompt('Literature search query:', 'PCI consciousness anesthesia');
      if (!q) return;
      const src = prompt('Source: pubmed or biorxiv', 'pubmed');
      const d = await (await fetch(BASE + '/literature/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, source: src || 'pubmed', max_results: 5 }),
      })).json();
      showResults('Literature: ' + q, d);

    } else if (cmd === 'lit-card') {
      const q = prompt('Evidence card query:', 'PCI anesthesia consciousness');
      if (!q) return;
      const d = await (await fetch(BASE + '/literature/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, source: 'pubmed', max_results: 3, as_card: true }),
      })).json();
      showResults('Evidence Cards: ' + q, d);

    } else if (cmd === 'linear-search') {
      const q = prompt('Linear search query:', 'awareness');
      if (q === null) return;
      const r = await fetch(BASE + '/linear/search?query=' + encodeURIComponent(q));
      const d = await r.json();
      showResults('Linear Issues', d);

    } else if (cmd === 'airtable-status') {
      const d = await (await fetch(BASE + '/airtable/status')).json();
      showResults('Airtable Status', d);

    } else if (cmd === 'airtable-sync') {
      const write = confirm('Allow write to Airtable? (Cancel = dry-run)');
      const r = await fetch(BASE + '/airtable/sync/runs?allow_write=' + write, { method: 'POST' });
      const d = await r.json();
      showResults('Airtable Sync', d);

    } else if (cmd === 'health') {
      const d = await (await fetch(BASE + '/health')).json();
      showResults('Health', d);
      await checkHealth();
    }
  } catch (err) {
    showResults('Error', { error: err.message });
  } finally {
    label.textContent = origText;
    // deactivate button after a moment so it's clear it ran
    setTimeout(() => btn.classList.remove('active'), 1500);
  }
}
