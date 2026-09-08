/**
 * Chat client for the AI Customer Support Agent demo.
 *
 * Keeps one session id, posts to /chat, and renders each reply with the tools
 * and knowledge-base articles behind it. Loading and error states are explicit,
 * and every status change is announced to assistive tech.
 */
(() => {
  "use strict";

  const el = {
    form: document.getElementById("chat-form"),
    input: document.getElementById("message-input"),
    send: document.getElementById("send-button"),
    thread: document.getElementById("messages"),
    empty: document.getElementById("empty-state"),
    customer: document.getElementById("customer-select"),
    reset: document.getElementById("reset-button"),
    banner: document.getElementById("status-banner"),
    bannerText: document.getElementById("status-text"),
    bannerIcon: document.querySelector("#status-banner use"),
    suggestions: document.getElementById("suggestions"),
    count: document.getElementById("char-count"),
  };

  const MAX_CHARS = 2000;
  const COUNT_VISIBLE_FROM = 1800;

  let sessionId = newSessionId();
  let pending = false;

  function newSessionId() {
    return `session_${Math.random().toString(36).slice(2, 12)}`;
  }

  function svgIcon(id, className = "icon") {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", className);
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#${id}`);
    svg.appendChild(use);
    return svg;
  }

  /** Show the status banner. `variant` is "error" | "info" | "warn". */
  function showBanner(text, variant = "error") {
    el.bannerText.textContent = text;
    el.banner.className = variant === "error" ? "banner" : `banner banner--${variant}`;
    el.bannerIcon.setAttribute("href", variant === "error" ? "#i-alert" : "#i-info");
    el.banner.hidden = false;
  }

  function hideBanner() {
    el.banner.hidden = true;
  }

  function clockTime() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  /**
   * Append a message.
   *
   * @param {"user"|"assistant"|"error"} role
   * @param {string} text
   * @param {{tools?: string[], sources?: string[]}} provenance
   */
  function addMessage(role, text, provenance = {}) {
    el.empty?.remove();

    const wrapper = document.createElement("article");
    wrapper.className = `msg msg--${role}`;

    const avatarIcon = { user: "i-user", error: "i-alert", assistant: "i-spark" }[role];
    const avatar = document.createElement("span");
    avatar.className = "msg__avatar";
    avatar.appendChild(svgIcon(avatarIcon));
    wrapper.appendChild(avatar);

    const body = document.createElement("div");
    body.className = "msg__body";

    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    bubble.textContent = text; // textContent, never innerHTML: no markup injection
    body.appendChild(bubble);

    const meta = document.createElement("div");
    meta.className = "msg__meta";

    const time = document.createElement("span");
    time.className = "msg__time";
    time.textContent = clockTime();
    meta.appendChild(time);

    for (const tool of provenance.tools ?? []) {
      meta.appendChild(makeChip("i-tool", tool, "chip chip--tool", `Tool called: ${tool}`));
    }
    for (const source of provenance.sources ?? []) {
      meta.appendChild(makeChip("i-doc", source, "chip", `Knowledge-base article: ${source}`));
    }

    body.appendChild(meta);
    wrapper.appendChild(body);
    el.thread.appendChild(wrapper);
    scrollToEnd();
    return wrapper;
  }

  function makeChip(iconId, label, className, title) {
    const chip = document.createElement("span");
    chip.className = className;
    chip.title = title;
    chip.appendChild(svgIcon(iconId));
    chip.appendChild(document.createTextNode(label));
    return chip;
  }

  function addTypingIndicator() {
    const wrapper = document.createElement("article");
    wrapper.className = "msg msg--assistant";

    const avatar = document.createElement("span");
    avatar.className = "msg__avatar";
    avatar.appendChild(svgIcon("i-spark"));

    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    const dots = document.createElement("span");
    dots.className = "dots";
    dots.append(
      document.createElement("span"),
      document.createElement("span"),
      document.createElement("span")
    );
    bubble.appendChild(dots);

    const body = document.createElement("div");
    body.className = "msg__body";
    body.appendChild(bubble);

    wrapper.append(avatar, body);
    el.thread.appendChild(wrapper);
    scrollToEnd();
    return wrapper;
  }

  function scrollToEnd() {
    el.thread.scrollTop = el.thread.scrollHeight;
  }

  function setPending(value) {
    pending = value;
    el.send.disabled = value;
    el.input.disabled = value;
    el.send.classList.toggle("is-busy", value);
    // aria-busy tells screen readers work is in flight without a visual-only cue.
    el.thread.setAttribute("aria-busy", String(value));
  }

  /** Grow the textarea with its content, up to the CSS max-height. */
  function autoGrow() {
    el.input.style.height = "auto";
    el.input.style.height = `${Math.min(el.input.scrollHeight, 148)}px`;
  }

  function updateCount() {
    const used = el.input.value.length;
    const show = used >= COUNT_VISIBLE_FROM;
    el.count.hidden = !show;
    if (show) {
      el.count.textContent = `${MAX_CHARS - used}`;
      el.count.classList.toggle("composer__count--limit", used >= MAX_CHARS);
    }
  }

  async function loadCustomers() {
    try {
      const response = await fetch("/api/customers");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const { customers } = await response.json();
      for (const customer of customers) {
        const option = document.createElement("option");
        option.value = customer.customer_id;
        option.textContent = `${customer.name} · ${customer.subscription}`;
        el.customer.appendChild(option);
      }
    } catch (error) {
      console.error("Could not load customers", error);
      showBanner("Couldn't load the demo customer list. Account questions may fail.", "warn");
    }
  }

  /** Pull a readable message out of an error response body. */
  async function readError(response) {
    try {
      const body = await response.json();
      return body.detail || body.error || `Request failed (${response.status}).`;
    } catch {
      return `Request failed (${response.status}).`;
    }
  }

  async function sendMessage(text) {
    const message = text.trim();
    if (pending || !message) return;

    hideBanner();
    addMessage("user", message);
    el.input.value = "";
    autoGrow();
    updateCount();
    setPending(true);
    const typing = addTypingIndicator();

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          customer_id: el.customer.value || null,
          session_id: sessionId,
        }),
      });

      typing.remove();

      if (!response.ok) {
        const detail = await readError(response);
        addMessage("error", detail);
        showBanner(detail, "error");
        return;
      }

      const data = await response.json();
      sessionId = data.session_id || sessionId;
      addMessage("assistant", data.answer, {
        tools: data.tools_used,
        sources: data.sources,
      });

      if (data.degraded) {
        showBanner("A backend service is unavailable — that answer is a fallback.", "warn");
      }
    } catch (error) {
      typing.remove();
      console.error(error);
      const detail = "Couldn't reach Trident. Check that the server is running.";
      addMessage("error", detail);
      showBanner(detail, "error");
    } finally {
      setPending(false);
      el.input.focus();
    }
  }

  el.form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(el.input.value);
  });

  // Enter sends; Shift+Enter inserts a newline. IME composition must not send.
  el.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage(el.input.value);
    }
  });

  el.input.addEventListener("input", () => {
    autoGrow();
    updateCount();
  });

  el.suggestions.addEventListener("click", (event) => {
    const prompt = event.target.closest("[data-prompt]")?.dataset.prompt;
    if (prompt) sendMessage(prompt);
  });

  el.reset.addEventListener("click", async () => {
    const previous = sessionId;
    sessionId = newSessionId();
    el.thread.replaceChildren();
    hideBanner();
    addMessage("assistant", "New conversation started. What can I help you with?");
    el.input.focus();
    try {
      await fetch(`/sessions/${previous}`, { method: "DELETE" });
    } catch {
      /* best effort: the session expires on its own anyway */
    }
  });

  el.customer.addEventListener("change", () => {
    if (!el.customer.value) {
      showBanner("Signed out. I can still answer general support questions.", "info");
      return;
    }
    const label = el.customer.selectedOptions[0].textContent;
    showBanner(`Signed in as ${label}. Account answers come from the customer API.`, "info");
  });

  loadCustomers();
  autoGrow();
  el.input.focus();
})();
