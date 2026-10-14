// The project page: shows recorded runs from recorded/runs.json, explains each one stage by
// stage (explain.js), and wakes the live demo.
(function () {
  "use strict";

  var root = document.documentElement;
  var $ = function (id) { return document.getElementById(id); };
  var explain = window.RagExplain;
  var systemPrompt = "";
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // ------------------------------------------------------------------ theme ---
  var toggle = $("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var dark = root.dataset.theme === "dark" ||
        (!root.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* ignore */ }
    });
  }

  // ------------------------------------------------------------- teaching ---
  if (explain) {
    explain.sandbox($("lab-body"));
    explain.concepts($("concepts-body"));
  }

  // ---------------------------------------------------------- recorded runs ---
  var STATUS_WORDS = {
    answered: "Answered with verified citations",
    insufficient_context: "Correctly declined: not in the documents",
    refused: "The model declined to answer"
  };
  function ms(value) {
    if (value === null || value === undefined) return "–";
    return value >= 1000 ? (value / 1000).toFixed(1) + " s" : Math.round(value) + " ms";
  }
  function median(values) {
    if (!values.length) return null;
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function renderRun(run) {
    var body = run.response;
    var card = $("run");
    card.textContent = "";
    card.appendChild(el("p", "run__why", run.why));
    card.appendChild(el("p", "run__q", body.query));
    var head = el("div", "run__head");
    var badge = el("span", "badge", STATUS_WORDS[body.status] || body.status);
    badge.dataset.status = body.status;
    head.appendChild(badge);
    head.appendChild(el("span", "run__meta",
      "confidence " + Number(body.confidence_score).toFixed(2) +
      " · retrieval " + ms(body.retrieval_latency_ms) +
      " · total " + ms(body.total_latency_ms) +
      (body.generation_model ? " · " + body.generation_model : "")));
    card.appendChild(head);
    card.appendChild(el("p", "run__answer", body.answer));

    card.appendChild(el("h3", null, "Verbatim citations"));
    var list = el("ol");
    if (!body.citations.length) {
      list.appendChild(el("li", "muted", "None: no citation is given for an answer the documents do not support."));
    }
    body.citations.forEach(function (c) {
      var li = el("li");
      li.appendChild(el("div", "where", c.document_title + " · page " + c.page_number));
      li.appendChild(el("blockquote", null, c.verbatim_quote));
      list.appendChild(li);
    });
    card.appendChild(list);

    var walk = el("section", "xp");
    walk.setAttribute("aria-label", "Walk through this answer");
    card.appendChild(walk);
    // The recorded response is kept exactly as the service returned it. A vector map replayed
    // afterwards from the same database (scripts/record_runs.py --maps-only) travels beside it, and
    // the walkthrough prefers it as the newer one; otherwise the response's own map is drawn.
    if (explain) explain.walkthrough(walk, body, { systemPrompt: systemPrompt, vectorMap: run.vector_map });
  }

  function showRecord(record) {
    var runs = record.runs || [];
    if (!runs.length) return;
    systemPrompt = record.system_prompt || "";
    if (systemPrompt) {
      $("rules-text").textContent = systemPrompt;
      $("rules").hidden = false;
    }
    var tabs = $("run-tabs");
    tabs.textContent = "";
    runs.forEach(function (run, index) {
      var tab = el("button", "tab", run.response.query);
      tab.type = "button";
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", index === 0 ? "true" : "false");
      tab.addEventListener("click", function () {
        Array.prototype.forEach.call(tabs.children, function (t) { t.setAttribute("aria-selected", "false"); });
        tab.setAttribute("aria-selected", "true");
        renderRun(run);
      });
      tabs.appendChild(tab);
    });
    renderRun(runs[0]);

    // Each recorded question says what it expects; the tiles count outcomes against that, so a
    // wrong refusal is never counted as a success.
    function tally(expected, boxId, valueId) {
      var graded = runs.filter(function (r) { return r.expect === expected; });
      if (!graded.length) { $(boxId).hidden = true; return; }
      var met = graded.filter(function (r) { return r.response.status === expected; });
      $(valueId).textContent = met.length + " of " + graded.length;
    }
    tally("answered", "tile-answered-box", "tile-answered");
    tally("insufficient_context", "tile-refused-box", "tile-refused");
    $("tile-retrieval").textContent = ms(median(runs.map(function (r) { return r.response.retrieval_latency_ms; })));
    $("tile-total").textContent = ms(median(runs.map(function (r) { return r.response.total_latency_ms; })));
    $("tiles").hidden = false;
    $("runs-note").hidden = false;

    var day = record.recorded_at ? record.recorded_at.slice(0, 10) : "";
    var chunks = (record.documents || []).reduce(function (n, d) { return n + d.chunks; }, 0);
    $("runs-meta").textContent = "Real answers from the service, saved as it returned them" +
      (day ? " on " + day : "") + ": " + (record.documents || []).length + " documents, " + chunks + " chunks.";
    $("foot-recorded").textContent = day ? "Runs recorded " + day : "";

    var stack = record.stack || {};
    if (stack.generator_chain) {
      $("stack-generator").textContent = "Free OpenRouter models, tried in order: " + stack.generator_chain.join(", then ");
    }
    if (stack.embeddings) $("stack-embeddings").textContent = "Local ONNX model " + stack.embeddings;
    if (stack.reranker) $("stack-reranker").textContent = "Local ONNX cross-encoder " + stack.reranker;
    if (stack.database) {
      $("stack-database").textContent = stack.database + ": HNSW (m=16, ef_construction=64) and tsvector (GIN), fused with RRF (k=" + stack.rrf_k + ")";
    }
  }

  fetch("recorded/runs.json", { headers: { "Accept": "application/json" } })
    .then(function (r) { if (!r.ok) throw new Error("runs " + r.status); return r.json(); })
    .then(showRecord)
    .catch(function () { /* the fallback sentence in the page stays */ });

  // --------------------------------------------------------------- live demo ---
  // The demo sleeps on a free host after 15 minutes without a request; the next request wakes
  // it in about a minute. It lives at another address, so this page may not read its answers.
  // What it can do is load the demo's icon as a picture: "load" fires only when the service
  // itself sends the icon, never for the host's own "waking up" page.
  var link = $("demo-link");
  var words = $("demo-words");
  var chip = $("demo-chip");
  if (!link || !words) return;

  var address;
  try { address = new URL(link.href); } catch (e) { address = null; }
  if (!address || address.protocol !== "https:" || /\.invalid$/.test(address.hostname)) {
    root.dataset.demo = "unset";
    // Not a link until the demo exists: no dead click, and nothing to reach with the keyboard.
    link.removeAttribute("href");
    link.setAttribute("aria-disabled", "true");
    words.textContent = "The live demo is not online yet. This page shows recorded runs instead.";
    return;
  }
  if (chip) { chip.href = address.href; chip.hidden = false; }

  var ICON = new URL("/playground/assets/favicon.svg", address).href;
  var RETRY_MS = 4000;
  var GIVE_UP_MS = 180000;
  var startedAt = Date.now();
  var done = false;

  function say(state, text) { root.dataset.demo = state; words.textContent = text; }

  function ask() {
    if (done) return;
    var image = new Image();
    var settled = false;
    var timer = setTimeout(function () { if (!settled) { settled = true; again(); } }, 20000);
    image.onload = function () {
      if (settled) return;
      settled = true; done = true; clearTimeout(timer);
      say("awake", "The live demo is awake. Ask it a question about the sample SOPs.");
    };
    image.onerror = function () { if (settled) return; settled = true; clearTimeout(timer); again(); };
    image.src = ICON + "?t=" + Date.now();
  }
  function again() {
    var waited = Date.now() - startedAt;
    if (waited > GIVE_UP_MS) {
      done = true;
      say("asleep", "The live demo has not answered for three minutes. The recorded runs below show what it does.");
      return;
    }
    if (waited > 4000) {
      say("waking", "Waking the live demo. It runs on free hosting and sleeps between visits, so this takes about a minute.");
    }
    setTimeout(ask, RETRY_MS);
  }
  ask();
})();
