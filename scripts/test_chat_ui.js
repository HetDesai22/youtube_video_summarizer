// Executes the page's <script> against a minimal DOM stub (no browser needed) and
// exercises the Chat with Video UI flow with a mocked fetch().
//   node scripts/test_chat_ui.js
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(
  path.join(__dirname, "..", "summarizer", "templates", "summarizer", "index.html"),
  "utf8"
);
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

class El {
  constructor(tag) {
    this.tagName = tag; this.children = []; this.listeners = {}; this.hidden = false;
    this.textContent = ""; this.value = ""; this.disabled = false; this.className = "";
    this.parent = null; this.scrollTop = 0; this.scrollHeight = 0; this._inner = "";
    this.href = ""; this.target = ""; this.rel = "";
    this.dataset = {};
    const self = this;
    this._classes = new Set();
    this.classList = {
      add: (c) => self._classes.add(c),
      remove: (c) => self._classes.delete(c),
      contains: (c) => self._classes.has(c),
      toggle: (c, force) => {
        const shouldAdd = force === undefined ? !self._classes.has(c) : force;
        shouldAdd ? self._classes.add(c) : self._classes.delete(c);
      },
    };
  }
  set innerHTML(v) { this._inner = v; if (v === "") this.children = []; }
  get innerHTML() { return this._inner; }
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((x) => x !== this); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  click() {
    // Mimics real DOM dispatch: listeners registered as `function(){ this... }`
    // (not arrow functions) rely on `this` being the element itself.
    for (const fn of this.listeners.click || []) fn.call(this, { preventDefault() {} });
  }
  querySelector(sel) {
    const cls = sel.slice(1);
    const walk = (n) => {
      for (const c of n.children) {
        if ((c.className || "").split(" ").includes(cls)) return c;
        const r = walk(c); if (r) return r;
      }
      return null;
    };
    return walk(this);
  }
  focus() {}
  reset() {}
  scrollIntoView() {}
  async fire(type) {
    for (const fn of this.listeners[type] || []) await fn({ preventDefault() {} });
  }
}

const elements = {};
const documentListeners = {};
const document = {
  cookie: "csrftoken=tok123; other=1",
  getElementById: (id) => (elements[id] = elements[id] || new El("div")),
  createElement: (tag) => new El(tag),
  addEventListener: (type, fn) => (documentListeners[type] = documentListeners[type] || []).push(fn),
};

const fetchCalls = [];
let fetchQueue = [];
const fetch = async (url, opts) => {
  fetchCalls.push({ url, opts });
  const next = fetchQueue.shift();
  if (next instanceof Error) throw next;
  return { json: async () => next };
};

// Fake timers: pollForQuestions schedules real multi-second delays, which
// tests shouldn't actually wait on. Scheduled callbacks are queued and only
// run when a test explicitly flushes them.
let pendingTimeouts = [];
const setTimeout = (fn, delay) => { pendingTimeouts.push(fn); return pendingTimeouts.length; };
async function flushOneTimeout() {
  const fn = pendingTimeouts.shift();
  if (fn) await fn();
}

const context = { document, fetch, encodeURIComponent, decodeURIComponent, JSON, Date, Math, setInterval, clearInterval, setTimeout, console };
vm.createContext(context);
vm.runInContext(script, context);

const $ = (id) => document.getElementById(id);
const text = (el) => el.children.map((c) => c.textContent + c.children.map((g) => g.textContent).join("|"));

(async () => {
  // --- summarize -> result shows chat section and questions discussed
  fetchQueue = [{
    success: true,
    data: {
      video_id: "abcDEF12345", chat_available: true, title: "T", channel: "C", duration: "1:00",
      thumbnail: "", summary: { summary: "S", main_topics: [], key_points: [], conclusions: [] },
      questions: [
        { id: 1, question: "What is machine learning?", start: 134.2, end: 142.7, timestamp: "02:14",
          youtube_url: "https://www.youtube.com/watch?v=abcDEF12345&t=134s" },
        { id: 2, question: "Why is training data important?", start: 337.4, end: 349.1, timestamp: "05:37",
          youtube_url: "https://www.youtube.com/watch?v=abcDEF12345&t=337s" },
      ],
    },
  }];
  $("youtube-url").value = "https://www.youtube.com/watch?v=abcDEF12345";
  await $("summarize-form").fire("submit");
  assert.strictEqual($("view-result").hidden, false, "result view shown");
  assert.strictEqual($("header-chat-btn").hidden, false, "header chat button shown when chat_available");
  assert.ok(!$("chat-panel").classList.contains("open"), "chat panel starts closed");

  assert.strictEqual($("questions-section").hidden, false, "questions section shown");
  assert.ok(!$("result-questions").classList.contains("show-all"), "grid starts collapsed");
  assert.strictEqual($("questions-toggle-btn").hidden, true, "toggle hidden when under the default cap");
  const qCards = $("result-questions").children;
  assert.strictEqual(qCards.length, 2, "two question cards rendered");
  const firstCard = qCards[0];
  const firstTop = firstCard.children.find((c) => c.className === "question-card-top");
  const firstLink = firstCard.children.find((c) => c.tagName === "a");
  assert.strictEqual(firstLink.href, "https://www.youtube.com/watch?v=abcDEF12345&t=134s");
  assert.strictEqual(firstLink.target, "_blank");
  const timeSpan = firstTop.children.find((c) => c.tagName === "span");
  const textP = firstLink.children.find((c) => c.tagName === "p");
  assert.strictEqual(timeSpan.textContent, "⏱ 02:14");
  assert.strictEqual(textP.textContent, "What is machine learning?");

  // Clicking "Ask AI" populates the chat input, opens the panel, and does
  // not itself call the API.
  const askButton = firstTop.children.find((c) => c.tagName === "button");
  assert.ok(askButton, "Ask AI button present when chat is available");
  const fetchCountBeforeAsk = fetchCalls.length;
  askButton.click();
  assert.strictEqual($("chat-input").value, "What is machine learning?");
  assert.ok($("chat-panel").classList.contains("open"), "Ask AI opens the chat panel");
  assert.strictEqual(fetchCalls.length, fetchCountBeforeAsk, "Ask AI does not itself call the API");

  // --- header button toggles the panel; close button and backdrop close it
  $("chat-panel").classList.remove("open");
  $("header-chat-btn").click();
  assert.ok($("chat-panel").classList.contains("open"), "header button opens the panel");
  $("chat-close-btn").click();
  assert.ok(!$("chat-panel").classList.contains("open"), "close button closes the panel");
  $("header-chat-btn").click();
  $("chat-backdrop").click();
  assert.ok(!$("chat-panel").classList.contains("open"), "clicking the backdrop closes the panel");
  $("header-chat-btn").click();
  documentListeners.keydown[0]({ key: "Escape" });
  assert.ok(!$("chat-panel").classList.contains("open"), "Escape key closes the panel");

  // --- successful chat turn with sources
  fetchQueue = [{
    success: true, answer: "Batteries store energy.", session_id: "sess-1",
    sources: [
      { label: "18:42 – 19:50", youtube_url: "https://www.youtube.com/watch?v=abcDEF12345&t=1122s" },
      { label: "evil", youtube_url: "javascript:alert(1)" },
    ],
  }];
  $("chat-input").value = "How is energy stored?";
  await $("chat-form").fire("submit");
  const call = fetchCalls[fetchCalls.length - 1];
  assert.strictEqual(call.url, "/chat/");
  assert.strictEqual(call.opts.method, "POST");
  assert.strictEqual(call.opts.headers["X-CSRFToken"], "tok123", "CSRF token sent from cookie");
  const body = JSON.parse(call.opts.body);
  assert.deepStrictEqual(body, { video_id: "abcDEF12345", session_id: null, message: "How is energy stored?" });

  const msgs = $("chat-messages").children;
  assert.strictEqual(msgs.length, 2, "user + assistant bubbles (hint and 'Thinking...' removed)");
  assert.ok(msgs[0].className.includes("user") && msgs[0].textContent === "How is energy stored?");
  assert.ok(msgs[1].className.includes("assistant"));
  const links = msgs[1].children[0].children.filter((c) => c.tagName === "a");
  assert.strictEqual(links.length, 1, "non-YouTube (javascript:) source link rejected");
  assert.strictEqual(links[0].href, "https://www.youtube.com/watch?v=abcDEF12345&t=1122s");
  assert.strictEqual(links[0].target, "_blank");
  assert.ok(links[0].rel.includes("noopener"));
  assert.ok(links[0].textContent.includes("18:42"));
  assert.strictEqual($("chat-send").disabled, false, "send re-enabled");

  // --- follow-up reuses the session id
  fetchQueue = [{ success: true, answer: "More.", sources: [], session_id: "sess-1" }];
  $("chat-input").value = "And then?";
  await $("chat-form").fire("submit");
  assert.strictEqual(JSON.parse(fetchCalls[fetchCalls.length - 1].opts.body).session_id, "sess-1");

  // --- server error shows an error bubble, not a crash
  fetchQueue = [{ success: false, error: "We couldn't generate an answer right now. Please try again." }];
  $("chat-input").value = "Q?";
  await $("chat-form").fire("submit");
  const last = $("chat-messages").children.pop();
  assert.ok(last.className.includes("error") && last.textContent.startsWith("We couldn't"));

  // --- network failure shows an error bubble
  fetchQueue = [new Error("offline")];
  $("chat-input").value = "Q2?";
  await $("chat-form").fire("submit");
  assert.ok($("chat-messages").children.pop().className.includes("error"));
  assert.strictEqual($("chat-send").disabled, false, "send re-enabled after failure");

  // --- empty input is ignored
  const before = fetchCalls.length;
  $("chat-input").value = "   ";
  await $("chat-form").fire("submit");
  assert.strictEqual(fetchCalls.length, before, "blank message not sent");

  // --- chat hidden when indexing failed; reset between videos
  fetchQueue = [{
    success: true,
    data: { video_id: "zzz", chat_available: false, title: "T", channel: "C", duration: "1:00",
            thumbnail: "", summary: { summary: "S" } },
  }];
  $("youtube-url").value = "https://www.youtube.com/watch?v=zzzzzzzzzzz";
  await $("summarize-form").fire("submit");
  assert.strictEqual($("header-chat-btn").hidden, true, "header button hidden when chat_available is false");
  assert.strictEqual($("questions-section").hidden, true, "no questions field -> section hidden");

  // --- questions shown without an "Ask AI" button when chat is unavailable
  // (the timestamp link itself never depends on chat).
  fetchQueue = [{
    success: true,
    data: {
      video_id: "noChatVideo1", chat_available: false, title: "T", channel: "C", duration: "1:00",
      thumbnail: "", summary: { summary: "S" },
      questions: [{ id: 1, question: "What is X?", start: 5.0, end: 9.0, timestamp: "00:05",
                    youtube_url: "https://www.youtube.com/watch?v=noChatVideo1&t=5s" }],
    },
  }];
  await $("summarize-form").fire("submit");
  assert.strictEqual($("questions-section").hidden, false, "questions shown even without chat");
  const noChatCard = $("result-questions").children[0];
  const noChatTop = noChatCard.children.find((c) => c.className === "question-card-top");
  assert.ok(!noChatTop.children.find((c) => c.tagName === "button"), "no Ask AI button without chat");
  assert.ok(noChatCard.children.find((c) => c.tagName === "a"), "timestamp link still present");

  fetchQueue = [{
    success: true,
    data: { video_id: "newVideo0001", chat_available: true, title: "T2", channel: "C", duration: "1:00",
            thumbnail: "", summary: { summary: "S2" } },
  }];
  await $("summarize-form").fire("submit");
  assert.strictEqual($("chat-messages").children.length, 1, "chat cleared (only hint) for a new video");
  fetchQueue = [{ success: true, answer: "A", sources: [], session_id: "sess-2" }];
  $("chat-input").value = "Q";
  await $("chat-form").fire("submit");
  const b = JSON.parse(fetchCalls[fetchCalls.length - 1].opts.body);
  assert.strictEqual(b.video_id, "newVideo0001");
  assert.strictEqual(b.session_id, null, "new video starts a new session");

  // --- more than 6 questions: extra cards start collapsed behind a toggle
  // (this is the fix for "the page is too long")
  const manyQuestions = Array.from({ length: 9 }, (_, i) => ({
    id: i + 1, question: "Question " + i + "?", start: i, end: i + 1,
    timestamp: "00:0" + i, youtube_url: "https://www.youtube.com/watch?v=manyQ&t=" + i + "s",
  }));
  fetchQueue = [{
    success: true,
    data: {
      video_id: "manyQ", chat_available: true, title: "Many", channel: "C", duration: "1:00",
      thumbnail: "", summary: { summary: "S" }, questions: manyQuestions,
    },
  }];
  await $("summarize-form").fire("submit");
  const allCards = $("result-questions").children;
  assert.strictEqual(allCards.length, 9, "all 9 cards are rendered (not truncated)");
  const extraCards = allCards.filter((c) => c.classList.contains("question-card-extra"));
  assert.strictEqual(extraCards.length, 3, "cards beyond the default 6 are marked extra");
  assert.ok(!allCards[0].classList.contains("question-card-extra"), "first 6 are not marked extra");
  assert.ok(!$("result-questions").classList.contains("show-all"), "grid collapsed by default");
  assert.strictEqual($("questions-toggle-btn").hidden, false, "toggle shown when over the cap");
  assert.strictEqual($("questions-toggle-btn").textContent, "Show all 9 questions");

  $("questions-toggle-btn").click();
  assert.ok($("result-questions").classList.contains("show-all"), "toggle reveals the extra cards");
  assert.strictEqual($("questions-toggle-btn").textContent, "Show fewer questions");

  $("questions-toggle-btn").click();
  assert.ok(!$("result-questions").classList.contains("show-all"), "toggle collapses again");
  assert.strictEqual($("questions-toggle-btn").textContent, "Show all 9 questions");

  // --- questions_pending: summary shows immediately with no questions yet;
  // a background poll (fake-timed) picks them up once they're ready.
  pendingTimeouts = [];
  fetchQueue = [{
    success: true,
    data: {
      video_id: "pendingVideo1", chat_available: true, title: "T3", channel: "C", duration: "2:00",
      thumbnail: "", summary: { summary: "S3", main_topics: [], key_points: [], conclusions: [] },
      questions: [], questions_pending: true,
    },
  }];
  await $("summarize-form").fire("submit");
  assert.strictEqual($("view-result").hidden, false, "summary shown without waiting for questions");
  assert.strictEqual($("questions-section").hidden, true, "no questions yet");
  assert.strictEqual(pendingTimeouts.length, 1, "one poll scheduled");

  // First poll: still not ready -> a second poll gets scheduled.
  fetchQueue = [{ success: true, data: { questions: [], chat_available: true } }];
  await flushOneTimeout();
  assert.strictEqual($("questions-section").hidden, true, "still no questions after first poll");
  assert.strictEqual(pendingTimeouts.length, 1, "a follow-up poll was scheduled");

  // Second poll: questions are ready now.
  fetchQueue = [{
    success: true,
    data: {
      chat_available: true,
      questions: [{ id: 1, question: "Ready now?", start: 1, end: 2, timestamp: "00:01",
                    youtube_url: "https://www.youtube.com/watch?v=pendingVideo1&t=1s" }],
    },
  }];
  await flushOneTimeout();
  assert.strictEqual($("questions-section").hidden, false, "questions appear once ready");
  const readyLink = $("result-questions").children[0].children.find((c) => c.tagName === "a");
  assert.strictEqual(readyLink.children[0].textContent, "Ready now?");
  assert.strictEqual(pendingTimeouts.length, 0, "polling stops once questions arrive");

  // --- a poll for a superseded (previous) video must not overwrite the new one
  pendingTimeouts = [];
  fetchQueue = [{
    success: true,
    data: {
      video_id: "supersededVid", chat_available: true, title: "Old", channel: "C", duration: "1:00",
      thumbnail: "", summary: { summary: "Old summary" }, questions: [], questions_pending: true,
    },
  }];
  await $("summarize-form").fire("submit");
  assert.strictEqual(pendingTimeouts.length, 1);

  // A different video is summarized before the old poll fires.
  fetchQueue = [{
    success: true,
    data: {
      video_id: "newerVid", chat_available: true, title: "New", channel: "C", duration: "1:00",
      thumbnail: "", summary: { summary: "New summary" }, questions: [], questions_pending: false,
    },
  }];
  await $("summarize-form").fire("submit");
  const staleFetchCountBefore = fetchCalls.length;
  await flushOneTimeout(); // fires the OLD video's stale poll
  assert.strictEqual(
    fetchCalls.length, staleFetchCountBefore,
    "a superseded poll is dropped before even calling the API"
  );
  assert.strictEqual($("questions-section").hidden, true, "no questions leaked in from the stale poll");
  assert.strictEqual($("result-title").textContent, "New", "still showing the newer video");

  console.log("UI chat flow: all assertions passed");
})().catch((e) => { console.error("FAILED:", e.message); process.exit(1); });
