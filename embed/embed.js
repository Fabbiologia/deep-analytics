/*
Embeddable chat widget for Deep Analytics agent.

Usage (on datamares.org):
  <script src="https://YOUR_HOST/static/embed.js" defer
          data-endpoint="https://YOUR_HOST/chat"
          data-title="Gulf of California Data Assistant"
          data-primary="#0e7c86"
          data-position="bottom-right"
  ></script>
  <link rel="stylesheet" href="https://YOUR_HOST/static/embed.css">

Alternatively, self-host embed.js and embed.css on the same server as embed_api.py
via a static files route.
*/
(function () {
  const script = document.currentScript;
  const ENDPOINT = script?.getAttribute('data-endpoint') || '/chat';
  const TITLE = script?.getAttribute('data-title') || 'Deep Analytics Assistant';
  const PRIMARY = script?.getAttribute('data-primary') || '#0e7c86';
  const POSITION = (script?.getAttribute('data-position') || 'bottom-right').toLowerCase();

  // Create container
  const container = document.createElement('div');
  container.id = 'da-embed-container';
  container.style.position = 'fixed';
  container.style.zIndex = 999999;
  container.style[POSITION.includes('right') ? 'right' : 'left'] = '20px';
  container.style[POSITION.includes('bottom') ? 'bottom' : 'top'] = '20px';
  document.body.appendChild(container);

  // Toggle button
  const btn = document.createElement('button');
  btn.id = 'da-embed-toggle';
  btn.setAttribute('aria-label', 'Open chat');
  btn.textContent = 'Chat';
  btn.style.background = PRIMARY;
  container.appendChild(btn);

  // Panel
  const panel = document.createElement('div');
  panel.id = 'da-embed-panel';
  panel.setAttribute('aria-hidden', 'true');
  panel.innerHTML = `
    <div class="da-embed-header">
      <div class="da-embed-title">${TITLE}</div>
      <button class="da-embed-close" aria-label="Close">×</button>
    </div>
    <div class="da-embed-body">
      <div class="da-embed-messages" role="log" aria-live="polite"></div>
      <div class="da-embed-input">
        <textarea placeholder="Ask about biomass, abundance, trends..." rows="2"></textarea>
        <button class="da-embed-send" aria-label="Send">Send</button>
      </div>
    </div>`;
  container.appendChild(panel);

  const closeBtn = panel.querySelector('.da-embed-close');
  const sendBtn = panel.querySelector('.da-embed-send');
  const textarea = panel.querySelector('textarea');
  const messages = panel.querySelector('.da-embed-messages');

  const setOpen = (open) => {
    panel.setAttribute('aria-hidden', open ? 'false' : 'true');
    container.classList.toggle('da-open', open);
    if (open) textarea?.focus();
  };

  btn.addEventListener('click', () => setOpen(!container.classList.contains('da-open')));
  closeBtn.addEventListener('click', () => setOpen(false));

  function appendMessage(text, role) {
    const el = document.createElement('div');
    el.className = `da-msg ${role}`;
    el.textContent = text;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
  }

  async function sendMessage() {
    const text = (textarea.value || '').trim();
    if (!text) return;
    appendMessage(text, 'user');
    textarea.value = '';
    appendMessage('Thinking…', 'assistant pending');

    try {
      const res = await fetch(ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const last = messages.querySelector('.da-msg.assistant.pending');
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        if (last) last.remove();
        appendMessage(`Error: ${err.detail || 'Request failed'}`, 'assistant error');
        return;
      }
      const data = await res.json();
      if (last) last.remove();
      appendMessage(data.answer || '(no answer)', 'assistant');
    } catch (e) {
      const last = messages.querySelector('.da-msg.assistant.pending');
      if (last) last.remove();
      appendMessage(`Network error: ${e}`, 'assistant error');
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  textarea.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  });
})();
