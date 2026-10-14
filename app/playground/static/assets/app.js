// The playground: one form, one API call, one honest result card, and the walkthrough that
// explains how that result was built (explain.js).
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var form = $("ask-form");
  var question = $("question");
  var filter = $("document-filter");
  var askBtn = $("ask-btn");
  var statusLine = $("status");
  var result = $("result");
  var chips = $("chips");
  var docs = $("docs");
  var explain = window.RagExplain;
  var systemPrompt = "";

  // ------------------------------------------------------------------ theme ---
  var toggle = $("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var root = document.documentElement;
      var dark = root.dataset.theme === "dark" ||
        (!root.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* ignore */ }
    });
  }

  // ---------------------------------------------------------------- helpers ---
  function setStatus(text, kind) {
    statusLine.textContent = text || "";
    if (kind) { statusLine.dataset.kind = kind; } else { delete statusLine.dataset.kind; }
  }
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function ms(value) { return value === null || value === undefined ? "n/a" : Math.round(value) + " ms"; }
  // Lists of numbers (the vector map's 384-number embeddings) on one line each, so the rest of
  // the raw response stays readable. Only real line breaks are joined: a JSON string never holds one.
  function rawJson(body) {
    return JSON.stringify(body, null, 2).replace(/\[\n\s+(-?[\d.eE+-]+(?:,\n\s+-?[\d.eE+-]+)*)\n\s*\]/g, function (match, inner) {
      return "[" + inner.replace(/,\n\s+/g, ", ") + "]";
    });
  }
  function fill(name, value) {
    if (value === null || value === undefined || value === "") return;
    Array.prototype.forEach.call(document.querySelectorAll('[data-fill="' + name + '"]'), function (node) {
      node.textContent = String(value);
    });
  }
  var STATUS_WORDS = {
    answered: "Answered with verified citations",
    insufficient_context: "The documents do not contain this",
    refused: "The model declined to answer"
  };

  // ------------------------------------------------------------- teaching ---
  if (explain) {
    explain.sandbox($("lab-body"));
    explain.concepts($("concepts-body"));
  }

  // ------------------------------------------------------------------ config ---
  function loadConfig() {
    return fetch("/playground/config", { headers: { "Accept": "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("config " + r.status); return r.json(); })
      .then(function (cfg) {
        if (cfg.demo_mode) { $("demo-note").hidden = false; }
        var retrieval = cfg.retrieval || {};
        fill("top_k_retrieval", retrieval.top_k_retrieval);
        fill("top_k_rerank", retrieval.top_k_rerank);
        if (retrieval.max_query_chars) question.maxLength = retrieval.max_query_chars;
        var models = cfg.models || {};
        fill("embedding", models.embedding);
        fill("reranker", models.reranker);
        if (cfg.system_prompt) {
          systemPrompt = cfg.system_prompt;
          $("rules-text").textContent = cfg.system_prompt;
          $("rules").hidden = false;
        }

        docs.textContent = "";
        if (!cfg.documents.length) {
          docs.appendChild(el("li", "muted", cfg.demo_mode
            ? "The corpus is being seeded; refresh in a minute."
            : "No documents yet. Ingest a PDF with POST /api/v1/ingest."));
        }
        cfg.documents.forEach(function (d) {
          var li = el("li");
          li.appendChild(el("span", "doc__title", d.title));
          li.appendChild(el("span", "doc__meta", d.pages_count + " pages · " + d.chunks_count + " chunks"));
          docs.appendChild(li);
          var opt = el("option", null, d.title);
          opt.value = d.id;
          filter.appendChild(opt);
        });
        chips.textContent = "";
        cfg.sample_questions.forEach(function (q) {
          var b = el("button", "chip", q);
          b.type = "button";
          b.addEventListener("click", function () { question.value = q; question.focus(); });
          chips.appendChild(b);
        });
        var gen = cfg.generator || {};
        $("foot-model").textContent = gen.model
          ? "Generation: " + gen.provider + " · " + gen.model
          : "Generation: not configured";
        var limits = cfg.limits || {};
        if (cfg.demo_mode && limits.per_client_per_minute) {
          var left = limits.free_requests_remaining_today;
          $("limits").textContent = "Limits: " + limits.per_client_per_minute + " questions per minute and "
            + limits.per_client_per_day + " per day for each visitor, "
            + limits.per_day + " per day for everyone"
            + (left !== null && left !== undefined ? " · free-model requests left today: " + left : "") + ".";
        }
      })
      .catch(function () {
        docs.textContent = "";
        docs.appendChild(el("li", "muted", "The service is waking up; refresh in a moment."));
      });
  }

  // ----------------------------------------------------------------- results ---
  function render(body) {
    result.hidden = false;
    var badge = $("result-badge");
    badge.dataset.status = body.status;
    badge.textContent = STATUS_WORDS[body.status] || body.status;
    $("result-meta").textContent = "confidence " + Number(body.confidence_score).toFixed(2)
      + " · retrieval " + ms(body.retrieval_latency_ms) + " · total " + ms(body.total_latency_ms)
      + (body.generation_model ? " · " + body.generation_model : "");
    $("result-answer").textContent = body.answer;
    var list = $("citations");
    list.textContent = "";
    if (!body.citations.length) {
      list.appendChild(el("li", "muted", body.is_grounded ? "No citations." : "No citation is given for an answer the documents do not support."));
    }
    body.citations.forEach(function (c) {
      var li = el("li");
      li.appendChild(el("div", "where", c.document_title + " · page " + c.page_number));
      li.appendChild(el("blockquote", null, c.verbatim_quote));
      list.appendChild(li);
    });
    // The API sends body.vector_map with the answer (null when the map is off or failed), and
    // the walkthrough draws it from there: nothing else is needed for it.
    if (explain) explain.walkthrough($("walkthrough"), body, { systemPrompt: systemPrompt });
    $("result-json").textContent = rawJson(body);
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function explainError(status, body) {
    var detail = body && typeof body.detail === "string" ? body.detail : "";
    if (status === 429) return detail || "Too many questions right now.";
    if (status === 503 || status === 502) return detail || "The service is temporarily unavailable; try again shortly.";
    if (status === 413) return "That question is far too long.";
    if (status === 422) {
      // A string comes from the demo's own checks (such as its length limit); a list comes from
      // schema validation of the request.
      return detail || "Please ask a question of 3 to " + question.maxLength + " characters, without control characters.";
    }
    return detail || ("Request failed (" + status + ").");
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var text = question.value.trim();
    if (text.length < 3) { setStatus("Please ask a question of at least three characters.", "error"); return; }
    askBtn.disabled = true;
    result.hidden = true;
    setStatus("Retrieving passages and writing a grounded answer… free models can take 10 to 40 seconds.");
    var payload = { query: text };
    if (filter.value) payload.filter_document_id = filter.value;
    var started = performance.now();
    fetch("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (r) {
        return r.json().catch(function () { return null; }).then(function (body) { return { status: r.status, body: body }; });
      })
      .then(function (res) {
        if (res.status === 200 && res.body) {
          setStatus("Answered in " + Math.round(performance.now() - started) + " ms round trip. The walkthrough under the answer explains each step.");
          render(res.body);
        } else {
          setStatus(explainError(res.status, res.body), "error");
        }
      })
      .catch(function () { setStatus("The request did not reach the service. Is it awake?", "error"); })
      .finally(function () { askBtn.disabled = false; });
  });

  loadConfig();
})();
