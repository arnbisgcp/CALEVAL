(function () {
  const chatStream = document.getElementById('chat-stream');
  const composerForm = document.getElementById('chat-composer-form');
  const messageInput = document.getElementById('chat-message-input');
  const sendBtn = document.getElementById('btn-send-message');
  const resetBtn = document.getElementById('btn-reset-session');
  const sessionBadge = document.getElementById('active-session-badge');
  const memberListEl = document.getElementById('member-directory-list');
  const claimListEl = document.getElementById('claim-directory-list');
  const paListEl = document.getElementById('pa-directory-list');

  let currentUserId = 'web_user_' + Math.random().toString(36).slice(2, 8);
  let currentSessionId = null;

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);
    // Bold **text**
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Inline code `code`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    const lines = html.split('\n');
    let out = [];
    let inList = false;

    for (let rawLine of lines) {
      const line = rawLine.trim();
      if (/^(\*|-|\d+\.)\s+/.test(line)) {
        if (!inList) {
          out.push('<ul>');
          inList = true;
        }
        const itemContent = line.replace(/^(\*|-|\d+\.)\s+/, '');
        out.push('<li>' + itemContent + '</li>');
      } else {
        if (inList) {
          out.push('</ul>');
          inList = false;
        }
        if (line.length > 0) {
          out.push('<p>' + line + '</p>');
        }
      }
    }
    if (inList) {
      out.push('</ul>');
    }
    return out.join('');
  }

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;
    row.appendChild(bubble);
    chatStream.appendChild(row);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  function appendLoadingBubble() {
    const row = document.createElement('div');
    row.className = 'message-row agent';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML =
      '<div class="loading-indicator"><span class="spinner"></span><span>Querying Vertex AI Reasoning Engine (6960836041880109056)...</span></div>';
    row.appendChild(bubble);
    chatStream.appendChild(row);
    chatStream.scrollTop = chatStream.scrollHeight;
    return row;
  }

  function appendAgentResponse(data) {
    const row = document.createElement('div');
    row.className = 'message-row agent';

    if (data.tool_calls && data.tool_calls.length > 0) {
      const tracesContainer = document.createElement('div');
      tracesContainer.className = 'tool-traces';

      data.tool_calls.forEach((tc, idx) => {
        const item = document.createElement('div');
        item.className = 'tool-trace-item';

        const header = document.createElement('div');
        header.className = 'tool-trace-header';

        const left = document.createElement('div');
        left.className = 'tool-trace-left';

        const badge = document.createElement('span');
        badge.className = 'tool-badge';
        badge.textContent = 'ADK Tool: ' + tc.name;

        const argsPreview = document.createElement('span');
        argsPreview.className = 'tool-args-preview';
        argsPreview.textContent = JSON.stringify(tc.args || {});

        left.appendChild(badge);
        left.appendChild(argsPreview);

        const toggleHint = document.createElement('span');
        toggleHint.className = 'data-card-code';
        toggleHint.textContent = 'Inspect JSON ▾';

        header.appendChild(left);
        header.appendChild(toggleHint);

        const body = document.createElement('div');
        body.className = 'tool-trace-body';
        body.textContent = JSON.stringify(
          {
            tool_call: tc.name,
            arguments: tc.args,
            response: tc.response || null,
          },
          null,
          2
        );

        header.addEventListener('click', () => {
          item.classList.toggle('open');
        });

        item.appendChild(header);
        item.appendChild(body);
        tracesContainer.appendChild(item);
      });

      row.appendChild(tracesContainer);
    }

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    const mdDiv = document.createElement('div');
    mdDiv.className = 'md-content';
    mdDiv.innerHTML = renderMarkdown(data.reply || 'No response returned.');
    bubble.appendChild(mdDiv);
    row.appendChild(bubble);

    chatStream.appendChild(row);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  async function sendPrompt(promptText) {
    const trimmed = (promptText || '').trim();
    if (!trimmed) return;

    messageInput.value = '';
    sendBtn.disabled = true;
    appendUserMessage(trimmed);
    const loadingRow = appendLoadingBubble();

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUserId,
          session_id: currentSessionId,
          message: trimmed,
        }),
      });
      const data = await resp.json();
      loadingRow.remove();

      if (data.session_id) {
        currentSessionId = data.session_id;
        sessionBadge.textContent = 'Session: ' + String(data.session_id).slice(-8);
      }

      if (!resp.ok || data.error) {
        appendAgentResponse({
          reply: '**Error communicating with Vertex AI Agent Engine:** ' + (data.error || resp.statusText),
          tool_calls: [],
        });
      } else {
        appendAgentResponse(data);
      }
    } catch (err) {
      loadingRow.remove();
      appendAgentResponse({
        reply: '**Network error:** ' + err.message,
        tool_calls: [],
      });
    } finally {
      sendBtn.disabled = false;
      messageInput.focus();
    }
  }

  composerForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendPrompt(messageInput.value);
  });

  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendPrompt(messageInput.value);
    }
  });

  resetBtn.addEventListener('click', () => {
    currentUserId = 'web_user_' + Math.random().toString(36).slice(2, 8);
    currentSessionId = null;
    sessionBadge.textContent = 'Session: New';
    const rows = chatStream.querySelectorAll('.message-row');
    rows.forEach((r) => r.remove());
  });

  document.querySelectorAll('.prompt-chip').forEach((btn) => {
    btn.addEventListener('click', () => {
      const prompt = btn.getAttribute('data-prompt');
      if (prompt) sendPrompt(prompt);
    });
  });

  async function loadMockDirectory() {
    try {
      const resp = await fetch('/api/mock-data');
      const data = await resp.json();

      // Render Members
      memberListEl.innerHTML = '';
      Object.values(data.members || {}).forEach((m) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'data-card';
        const statusClass = 'badge-' + String(m.status).toLowerCase();
        btn.innerHTML = `
          <div class="data-card-top">
            <span class="data-card-title">${escapeHtml(m.full_name)}</span>
            <span class="badge ${statusClass}">${escapeHtml(m.status)}</span>
          </div>
          <div class="data-card-sub">
            <span class="data-card-code">${escapeHtml(m.member_id)}</span>
            <span>${escapeHtml(m.plan_name)}</span>
          </div>
        `;
        btn.addEventListener('click', () => {
          sendPrompt(
            `Check member eligibility, deductible/out-of-pocket accumulators, and list all claims for ${m.full_name} (${m.member_id}).`
          );
        });
        memberListEl.appendChild(btn);
      });

      // Render Claims
      claimListEl.innerHTML = '';
      Object.values(data.claims || {}).forEach((c) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'data-card';
        const statusClass = 'badge-' + String(c.status).toLowerCase();
        btn.innerHTML = `
          <div class="data-card-top">
            <span class="data-card-code">${escapeHtml(c.claim_id)}</span>
            <span class="badge ${statusClass}">${escapeHtml(c.status)}</span>
          </div>
          <div class="data-card-sub">
            <span>${escapeHtml(c.patient_name)}</span>
            <span class="data-card-code">$${Number(c.total_billed).toLocaleString()}</span>
          </div>
        `;
        btn.addEventListener('click', () => {
          sendPrompt(
            `Get full claim details for ${c.claim_id} (${c.patient_name}), explain the line items and adjudication status, and recommend next steps.`
          );
        });
        claimListEl.appendChild(btn);
      });

      // Render Prior Authorizations
      paListEl.innerHTML = '';
      Object.values(data.prior_auths || {}).forEach((pa) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'data-card';
        const statusClass = 'badge-' + String(pa.status).toLowerCase();
        btn.innerHTML = `
          <div class="data-card-top">
            <span class="data-card-code">${escapeHtml(pa.pa_id)}</span>
            <span class="badge ${statusClass}">${escapeHtml(pa.status)}</span>
          </div>
          <div class="data-card-sub">
            <span>${escapeHtml(pa.patient_name)}</span>
            <span class="data-card-code">CPT ${escapeHtml(pa.cpt_code)}</span>
          </div>
        `;
        btn.addEventListener('click', () => {
          sendPrompt(
            `Check prior authorization ${pa.pa_id} for ${pa.patient_name} (CPT ${pa.cpt_code}) and explain its status and notes.`
          );
        });
        paListEl.appendChild(btn);
      });
    } catch (err) {
      console.error('Failed to load mock directory:', err);
    }
  }

  loadMockDirectory();
})();
