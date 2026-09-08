/**
 * Minimal chat client for the AI Customer Support Agent demo.
 *
 * Responsibilities: keep one session id, post to /chat, render messages with
 * the tools the agent used, and show explicit loading and error states.
 */
(() => {
  "use strict";

  const el = {
    form: document.getElementById("chat-form"),
    input: document.getElementById("message-input"),
    send: document.getElementById("send-button"),
    messages: document.getElementById("messages"),
    customer: document.getElementById("customer-select"),
    reset: document.getElementById("reset-button"),
    banner: document.getElementById("status-banner"),
    suggestions: document.getElementById("suggestions"),
  };

  let sessionId = newSessionId();
  let pending = false;

  function newSessionId() {
    return `session_${Math.random().toString(36).slice(2, 12)}`;
  }

  function showBanner(text, variant = "error") {
    el.banner.textContent = text;
    el.banner.className = variant === "info" ? "banner banner--info" : "banner";
    el.banner.hidden = false;
  }

  function hideBanner() {
    el.banner.hidden = true;
  }

  /** Append a message bubble. `tags` render as small chips under the text. */
  function addMessage(role, text, tags = []) {
    const article = document.createElement("article");
    article.className = `message message--${role}`;

    const bubble = document.createElement("div");
    bubble.className = "message__bubble";
    bubble.textContent = text; // textContent, never innerHTML: no markup injection
    article.appendChild(bubble);

    if (tags.length) {
      const meta = document.createElement("div");
      meta.className = "message__meta";
      for (const tag of tags) {
        const chip = document.createElement("span");
        chip.className = tag.kind === "source" ? "tag tag--source" : "tag";
        chip.textContent = tag.label;
        meta.appendChild(chip);
      }
      article.appendChild(meta);
    }

    el.messages.appendChild(article);
    el.messages.scrollTop = el.messages.scrollHeight;
    return article;
  }

  function addTypingIndicator() {
    const article = document.createElement("article");
    article.className = "message message--assistant";
    article.innerHTML =
      '<div class="message__bubble"><span class="dots"><span></span><span></span><span></span></span></div>';
    el.messages.appendChild(article);
    el.messages.scrollTop = el.messages.scrollHeight;
    return article;
  }

  function setPending(value) {
    pending = value;
    el.send.disabled = value;
    el.input.disabled = value;
    el.send.querySelector(".button__label").textContent = value ? "Sending…" : "Send";
  }

  /** Populate the customer selector from the mock backend. */
  async function loadCustomers() {
    try {
      const response = await fetch("/api/customers");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const { customers } = await response.json();
      for (const customer of customers) {
        const option = document.createElement("option");
        option.value = customer.customer_id;
        option.textContent = `${customer.name} · ${customer.subscription} (${customer.customer_id})`;
        el.customer.appendChild(option);
      }
    } catch (error) {
      console.error("Could not load customers", error);
      showBanner("Could not load the demo customer list. Account questions may fail.");
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
    if (pending || !text.trim()) return;

    hideBanner();
    addMessage("user", text);
    el.input.value = "";
    setPending(true);
    const typing = addTypingIndicator();

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          customer_id: el.customer.value || null,
          session_id: sessionId,
        }),
      });

      typing.remove();

      if (!response.ok) {
        const detail = await readError(response);
        addMessage("error", detail);
        showBanner(detail);
        return;
      }

      const data = await response.json();
      sessionId = data.session_id || sessionId;

      const tags = [
        ...(data.tools_used || []).map((name) => ({ kind: "tool", label: `🔧 ${name}` })),
        ...(data.sources || []).map((id) => ({ kind: "source", label: `📄 ${id}` })),
      ];
      addMessage("assistant", data.answer, tags);

      if (data.degraded) {
        showBanner("A backend service is unavailable — that answer is a fallback.", "info");
      }
    } catch (error) {
      typing.remove();
      console.error(error);
      const detail = "Could not reach the assistant. Check that the server is running.";
      addMessage("error", detail);
      showBanner(detail);
    } finally {
      setPending(false);
      el.input.focus();
    }
  }

  el.form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(el.input.value);
  });

  el.suggestions.addEventListener("click", (event) => {
    const prompt = event.target.dataset?.prompt;
    if (prompt) sendMessage(prompt);
  });

  el.reset.addEventListener("click", async () => {
    const previous = sessionId;
    sessionId = newSessionId();
    el.messages.replaceChildren();
    hideBanner();
    addMessage("assistant", "New conversation started. What can I help you with?");
    // Best-effort: free the old session server-side.
    try {
      await fetch(`/sessions/${previous}`, { method: "DELETE" });
    } catch {
      /* the session expires on its own anyway */
    }
  });

  el.customer.addEventListener("change", () => {
    const label = el.customer.selectedOptions[0]?.textContent ?? "Not signed in";
    showBanner(`Signed in as ${label}. Account answers come from the customer API.`, "info");
  });

  loadCustomers();
  el.input.focus();
})();
