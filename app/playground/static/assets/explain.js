// Teaching layer shared by the project page (site/) and the playground (app/playground/static/).
//
//   RagExplain.walkthrough(node, response, context)  every stage of one answer, with its numbers
//   RagExplain.sandbox(node)                          a rank-fusion lab with a k slider
//   RagExplain.concepts(node)                         the glossary, each idea linked to its code
//
// The two copies of this file must stay identical (tests/test_site.py compares them). Everything
// is built with createElement and textContent: answers and documents are never parsed as HTML.
(function () {
  "use strict";

  var REPO = "https://github.com/nickdesjardins/hybrid-knowledge-rag/blob/main/";

  // ---------------------------------------------------------------- helpers ---
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }
  // A paragraph from strings and nodes; strings wrapped in [[...]] become <code>.
  function para(cls, parts) {
    var p = el("p", cls);
    parts.forEach(function (part) {
      if (typeof part !== "string") { p.appendChild(part); return; }
      part.split(/(\[\[[^\]]+\]\])/).forEach(function (piece) {
        if (!piece) return;
        var m = /^\[\[([^\]]+)\]\]$/.exec(piece);
        p.appendChild(m ? el("code", null, m[1]) : document.createTextNode(piece));
      });
    });
    return p;
  }
  function ms(v) {
    if (v === null || v === undefined) return "n/a";
    return v >= 1000 ? (v / 1000).toFixed(1) + " s" : Math.max(1, Math.round(v)) + " ms";
  }
  function plural(n, one, many) { return n + " " + (n === 1 ? one : (many || one + "s")); }
  function shorten(text, max) { return text.length > max ? text.slice(0, max - 1) + "…" : text; }
  function codeLinks(paths) {
    var p = el("p", "xp-code");
    p.appendChild(document.createTextNode("Read the code: "));
    paths.forEach(function (path, i) {
      if (i) p.appendChild(document.createTextNode(", "));
      var a = el("a", null, path);
      a.href = REPO + path;
      a.rel = "noopener";
      p.appendChild(a);
    });
    return p;
  }
  function table(headers, rows, cls) {
    var wrap = el("div", "tablewrap");
    var t = el("table", "xp-table" + (cls ? " " + cls : ""));
    var thead = el("thead");
    var tr = el("tr");
    headers.forEach(function (h) { tr.appendChild(el("th", null, h)); });
    thead.appendChild(tr);
    t.appendChild(thead);
    var tbody = el("tbody");
    rows.forEach(function (cells) {
      var row = el("tr");
      cells.forEach(function (cell) {
        var td = el("td", cell && cell.num ? "num" : null);
        if (cell && cell.node) td.appendChild(cell.node);
        else td.textContent = cell && cell.text !== undefined ? cell.text : String(cell);
        row.appendChild(td);
      });
      tbody.appendChild(row);
    });
    t.appendChild(tbody);
    wrap.appendChild(t);
    return { wrap: wrap, tbody: tbody };
  }
  function passageCell(p) {
    var box = el("div");
    box.appendChild(el("div", "where", shorten(p.document_title, 60) + " · page " + p.page_number));
    box.appendChild(el("div", "preview", shorten(p.preview || "", 110)));
    return { node: box };
  }
  // A horizontal bar split into labelled parts (CSSOM sizing is allowed by the page's CSP).
  function stackedBar(parts, caption) {
    var fig = el("figure", "xp-stack");
    var bar = el("div", "xp-bar");
    var total = parts.reduce(function (n, part) { return n + part.value; }, 0) || 1;
    parts.forEach(function (part) {
      if (!part.value) return;
      var seg = el("span", "xp-seg xp-seg--" + part.key);
      seg.style.flexGrow = String(Math.max(part.value, total * 0.015));
      seg.title = part.label + ": " + part.text;
      bar.appendChild(seg);
    });
    fig.appendChild(bar);
    var legend = el("figcaption", "xp-legend");
    parts.forEach(function (part) {
      var item = el("span", "xp-legend__item");
      item.appendChild(el("span", "xp-dot xp-seg--" + part.key));
      item.appendChild(document.createTextNode(part.label + " " + part.text));
      legend.appendChild(item);
    });
    fig.appendChild(legend);
    if (caption) fig.appendChild(el("p", "xp-note", caption));
    return fig;
  }

  // ------------------------------------------------------------ walkthrough ---
  function step(n, title, summary, timing) {
    var li = el("li", "xp-step");
    var details = el("details");
    var summaryNode = el("summary");
    summaryNode.appendChild(el("span", "xp-n", String(n)));
    var head = el("span", "xp-head");
    head.appendChild(el("span", "xp-name", title));
    head.appendChild(el("span", "xp-sum", summary));
    summaryNode.appendChild(head);
    if (timing !== null && timing !== undefined) summaryNode.appendChild(el("span", "xp-time", ms(timing)));
    details.appendChild(summaryNode);
    var body = el("div", "xp-body");
    details.appendChild(body);
    li.appendChild(details);
    return { li: li, body: body, details: details };
  }
  function why(text) { return para("xp-why", [el("strong", null, "Why: "), text]); }

  function rrfFormula(k, p) {
    var terms = [];
    if (p.dense_rank !== null && p.dense_rank !== undefined) terms.push({ label: "1/(" + k + "+" + p.dense_rank + ")", value: 1 / (k + p.dense_rank) });
    if (p.sparse_rank !== null && p.sparse_rank !== undefined) terms.push({ label: "1/(" + k + "+" + p.sparse_rank + ")", value: 1 / (k + p.sparse_rank) });
    if (!terms.length) return "not found by either arm";
    var sum = terms.reduce(function (n, t) { return n + t.value; }, 0);
    var text = terms.map(function (t) { return t.label; }).join(" + ") + " = " +
      terms.map(function (t) { return t.value.toFixed(5); }).join(" + ");
    return terms.length > 1 ? text + " = " + sum.toFixed(5) : text;
  }

  function walkthrough(container, r, context) {
    context = context || {};
    container.textContent = "";
    var p = r && r.pipeline;
    if (!p) { container.hidden = true; return; }
    container.hidden = false;
    var passages = r.retrieved_passages || [];
    var k = p.rrf_k;
    var t = p.timings || {};

    container.appendChild(el("h3", "xp-title", "Walk through this answer"));
    container.appendChild(el("p", "xp-lede",
      "Every number below comes from this exact answer. Open a step to see what happened, why the step exists, and where it lives in the code."));

    var timeParts = [
      { key: "embed", label: "Embed", value: t.embed_ms || 0, text: ms(t.embed_ms) },
      { key: "search", label: "Search", value: t.search_ms || 0, text: ms(t.search_ms) },
      { key: "rerank", label: "Re-rank", value: t.rerank_ms || 0, text: ms(t.rerank_ms) }
    ];
    if (t.generate_ms !== null && t.generate_ms !== undefined) {
      timeParts.push({ key: "generate", label: "Generate + verify", value: t.generate_ms, text: ms(t.generate_ms) });
    }
    container.appendChild(stackedBar(timeParts,
      "Where the time went. Retrieval runs on the server itself in milliseconds; the language model is a call over the network to a shared free model, so it usually dominates."));

    var toggle = el("button", "xp-toggle", "Open all steps");
    toggle.type = "button";
    container.appendChild(toggle);

    var list = el("ol", "xp-steps");
    var all = [];

    // 1. Embed -------------------------------------------------------------
    var s1 = step(1, "Turn the question into a vector", plural(p.embedding_dimensions, "number"), t.embed_ms);
    s1.body.appendChild(para(null, [
      "The embedding model [[" + p.embedding_model + "]] turned your question into a list of " +
      p.embedding_dimensions + " numbers. Every stored chunk was turned into such a list when it was ingested. " +
      "Texts that mean similar things get lists that point in similar directions, even with no word in common."
    ]));
    s1.body.appendChild(why("it lets the search match “keep” with “retain”, which keyword search cannot do."));
    s1.body.appendChild(codeLinks(["app/retrieval/embedder.py"]));
    all.push(s1);

    // 2. Search -----------------------------------------------------------
    var s2 = step(2, "Search two ways in one SQL statement",
      plural(p.candidates, "candidate") + ": " + p.found_by_both + " by both, " +
      p.found_by_vector_only + " vector only, " + p.found_by_keyword_only + " keyword only", t.search_ms);
    s2.body.appendChild(para(null, [
      "PostgreSQL ran two searches at once. The vector arm walked an HNSW index to find the chunks whose vectors are closest to the question (cosine distance, operator [[<=>]]). " +
      "The keyword arm matched the question's words against a full-text index ([[tsvector]] with a GIN index). " +
      "Each arm could return up to " + p.candidate_limit + " chunks; together they found " + plural(p.candidates, "distinct chunk") + "."
    ]));
    s2.body.appendChild(stackedBar([
      { key: "both", label: "Found by both", value: p.found_by_both, text: String(p.found_by_both) },
      { key: "vector", label: "Vector only", value: p.found_by_vector_only, text: String(p.found_by_vector_only) },
      { key: "keyword", label: "Keyword only", value: p.found_by_keyword_only, text: String(p.found_by_keyword_only) }
    ]));
    s2.body.appendChild(why("each arm misses what the other catches. Embeddings blur exact identifiers such as “21 CFR Part 11”; keywords miss synonyms and paraphrases."));
    s2.body.appendChild(codeLinks(["app/retrieval/hybrid_search.py", "app/database.py"]));
    all.push(s2);

    // 3. Fuse -------------------------------------------------------------
    var s3 = step(3, "Fuse the two rankings (RRF)", "k = " + k + ", scores computed from ranks only", null);
    s3.body.appendChild(para(null, [
      "A cosine distance and a keyword relevance score live on different scales, so they cannot simply be added. " +
      "Reciprocal Rank Fusion ignores the scores and uses positions: a chunk earns [[1 / (k + rank)]] from every list it appears in, and the sums give the fused order. " +
      "Here is the arithmetic for the passages that reached the model:"
    ]));
    if (passages.length) {
      s3.body.appendChild(table(
        ["Passage", "Vector rank", "Keyword rank", "1 / (k + rank), summed", "Fused position"],
        passages.map(function (q) {
          return [
            passageCell(q),
            { text: q.dense_rank ? "#" + q.dense_rank : "–", num: true },
            { text: q.sparse_rank ? "#" + q.sparse_rank : "–", num: true },
            { text: rrfFormula(k, q), num: true },
            { text: q.fused_rank ? "#" + q.fused_rank + " of " + p.candidates : "–", num: true }
          ];
        })).wrap);
    }
    s3.body.appendChild(why("a chunk that both arms rank fairly well beats one that a single arm ranks first, and no score calibration is needed. Try other values of k in the rank-fusion lab on this page."));
    s3.body.appendChild(codeLinks(["app/retrieval/hybrid_search.py"]));
    all.push(s3);

    // 4. Re-rank ----------------------------------------------------------
    var s4 = step(4, "Re-rank with a cross-encoder",
      "kept " + p.passages_to_model + " of " + plural(p.candidates, "candidate"), t.rerank_ms);
    s4.body.appendChild(para(null, [
      "The cross-encoder [[" + p.reranker_model + "]] read the question and each fused candidate together, as one input, and scored how well the passage answers it. " +
      "The best " + p.passages_to_model + " went on to the model."
    ]));
    if (passages.length) {
      s4.body.appendChild(table(
        ["Passage", "Fused position", "Re-ranked position", "Cross-encoder score", "Cited"],
        passages.map(function (q) {
          return [
            passageCell(q),
            { text: q.fused_rank ? "#" + q.fused_rank : "–", num: true },
            { text: "#" + q.rerank_position, num: true },
            { text: Number(q.relevance_score).toFixed(2), num: true },
            { text: q.cited ? "yes" : "", num: true }
          ];
        })).wrap);
      var lifted = passages.filter(function (q) { return q.fused_rank && q.fused_rank > q.rerank_position; })
        .sort(function (a, b) { return (b.fused_rank - b.rerank_position) - (a.fused_rank - a.rerank_position); })[0];
      s4.body.appendChild(el("p", "xp-note", lifted
        ? "The re-ranker lifted a passage from fused position #" + lifted.fused_rank + " to #" + lifted.rerank_position + ": it noticed something rank fusion could not."
        : "Here the re-ranker agreed with the fused order."));
    }
    s4.body.appendChild(why("the embedding model reads question and passage separately and compares two summaries. A cross-encoder reads them side by side, which is slower but much sharper, so it only runs on the short list. Its scores are raw logits: higher is better, and negative values are normal."));
    s4.body.appendChild(codeLinks(["app/retrieval/reranker.py"]));
    all.push(s4);

    // 5. Generate ---------------------------------------------------------
    var model = r.generation_model || "the language model";
    var s5 = step(5, "Answer under a JSON schema", r.generation_model ? shorten(r.generation_model, 48) : "no model call", t.generate_ms);
    s5.body.appendChild(para(null, [
      "Only these " + plural(p.passages_to_model, "passage") + " were sent to [[" + model + "]], each wrapped in tags that carry a random suffix, so text inside a document cannot pose as an instruction. " +
      "The model had to reply with JSON matching a schema: the answer, whether it is grounded, a confidence score, and quotes, each tied to a passage id."
    ]));
    if (context.systemPrompt) {
      var rules = el("details", "xp-prompt");
      rules.appendChild(el("summary", null, "Read the rules the model answers under"));
      rules.appendChild(el("pre", null, context.systemPrompt));
      s5.body.appendChild(rules);
    }
    s5.body.appendChild(why("a schema makes the reply machine-checkable, which is what makes the next step possible. Three passages instead of fifty also keep the model from losing the important one in the middle of a long prompt."));
    s5.body.appendChild(codeLinks(["app/generator/grounded_llm.py", "app/openrouter.py"]));
    all.push(s5);

    // 6. Verify -----------------------------------------------------------
    var proposed = p.quotes_proposed;
    var verified = p.quotes_verified;
    var verdict;
    if (r.status === "answered") verdict = verified + " of " + plural(proposed, "quote") + " matched the document";
    else if (r.status === "refused") verdict = "the model declined to answer";
    else if (proposed) verdict = "no quote matched, so the draft was withheld";
    else verdict = "the documents do not answer this";
    var s6 = step(6, "Verify every quote", verdict, null);
    if (r.status === "answered") {
      s6.body.appendChild(para(null, [
        "The model proposed " + plural(proposed, "quote") + ". The server searched for each one in the stored text of the passage it names, ignoring only differences of case, spacing and typography, and kept " + verified + ". " +
        "Each citation's page comes from that chunk's page map: the page on which the quote begins. The model never supplies a page number."
      ]));
    } else if (proposed) {
      s6.body.appendChild(para(null, [
        "The model proposed " + plural(proposed, "quote") + ", but none of them could be found in the stored passages. An answer without a verifiable quote is not shown."
      ]));
    } else {
      s6.body.appendChild(para(null, [
        "The model reported that these passages do not contain the answer. The engine returns the status [[" + r.status + "]] with no citations instead of guessing."
      ]));
    }
    s6.body.appendChild(why("models misquote numbers and invent references even when told not to. Checking every quote against the source turns “trust me” into something an auditor can check."));
    s6.body.appendChild(codeLinks(["app/generator/grounded_llm.py", "app/ingestion/chunker.py"]));
    all.push(s6);

    all.forEach(function (s) { list.appendChild(s.li); });
    container.appendChild(list);

    toggle.addEventListener("click", function () {
      var open = all.some(function (s) { return !s.details.open; });
      all.forEach(function (s) { s.details.open = open; });
      toggle.textContent = open ? "Close all steps" : "Open all steps";
    });
  }

  // ---------------------------------------------------------- rank-fusion lab ---
  var LAB = {
    question: "Who approves the closure of a critical deviation?",
    items: [
      { id: "A", text: "Critical deviations: closure requires approval by the Head of Quality.", vector: 3, keyword: 2, answer: true },
      { id: "B", text: "Minor and major deviations: closure is approved by the QA Manager.", vector: 1, keyword: 5 },
      { id: "C", text: "Form QA-023-F4: Critical Deviation Closure Record.", vector: null, keyword: 1 },
      { id: "D", text: "Deviation records are retained for seven years.", vector: 2, keyword: null },
      { id: "E", text: "CAPA effectiveness checks are approved by QA.", vector: 4, keyword: 3 },
      { id: "F", text: "Deviation trends are reported to management every quarter.", vector: 5, keyword: 4 }
    ]
  };
  var labCount = 0;

  function sandbox(container) {
    container.textContent = "";
    labCount += 1;
    var uid = "xp-lab-" + labCount;
    container.appendChild(para("xp-lede", [
      "Six passages, each ranked by the two search arms for one question (a dash means that arm did not return it). " +
      "Change k or switch an arm off and watch the fused order. Passage A is the one that actually answers the question."
    ]));
    container.appendChild(el("p", "xp-question", "“" + LAB.question + "”"));

    var controls = el("div", "xp-controls");
    var kLabel = el("label", "xp-k");
    kLabel.htmlFor = uid + "-k";
    kLabel.appendChild(document.createTextNode("k = "));
    var kOut = el("output", null, "60");
    kOut.htmlFor = uid + "-k";
    kLabel.appendChild(kOut);
    var kInput = el("input");
    kInput.type = "range"; kInput.min = "1"; kInput.max = "100"; kInput.step = "1"; kInput.value = "60";
    kInput.id = uid + "-k";
    controls.appendChild(kLabel);
    controls.appendChild(kInput);
    function checkbox(label) {
      var wrap = el("label", "xp-check");
      var box = el("input");
      box.type = "checkbox"; box.checked = true;
      wrap.appendChild(box);
      wrap.appendChild(document.createTextNode(" " + label));
      controls.appendChild(wrap);
      return box;
    }
    var useVector = checkbox("Vector search");
    var useKeyword = checkbox("Keyword search");
    container.appendChild(controls);

    var grid = table(["Fused", "Passage", "Vector rank", "Keyword rank", "1 / (k + rank), summed"], [], "xp-lab");
    container.appendChild(grid.wrap);
    var note = el("p", "xp-note");
    note.setAttribute("role", "status");
    container.appendChild(note);

    function fuse(k, vectorOn, keywordOn) {
      return LAB.items.map(function (item) {
        var v = vectorOn ? item.vector : null;
        var s = keywordOn ? item.keyword : null;
        var terms = [];
        if (v) terms.push({ label: "1/(" + k + "+" + v + ")", value: 1 / (k + v) });
        if (s) terms.push({ label: "1/(" + k + "+" + s + ")", value: 1 / (k + s) });
        var score = terms.reduce(function (n, t) { return n + t.value; }, 0);
        var formula = terms.length
          ? terms.map(function (t) { return t.label; }).join(" + ") + " = " + score.toFixed(5)
          : "not returned";
        return { item: item, v: v, s: s, score: score, formula: formula };
      }).filter(function (row) { return row.score > 0; })
        .sort(function (a, b) { return b.score - a.score || (a.item.id < b.item.id ? -1 : 1); });
    }
    // The smallest k at which the answer comes first with both arms on (computed, not claimed).
    var tip = null;
    for (var trial = 1; trial <= 100 && tip === null; trial++) {
      if (fuse(trial, true, true)[0].item.answer) tip = trial;
    }

    function render() {
      var k = Number(kInput.value);
      kOut.textContent = String(k);
      var scored = fuse(k, useVector.checked, useKeyword.checked);

      grid.tbody.textContent = "";
      scored.forEach(function (row, index) {
        var tr = el("tr", row.item.answer ? "xp-answer" : null);
        tr.appendChild(el("td", "num", "#" + (index + 1)));
        var cell = el("td");
        cell.appendChild(el("strong", null, row.item.id + " "));
        cell.appendChild(document.createTextNode(row.item.text));
        tr.appendChild(cell);
        tr.appendChild(el("td", "num", row.v ? "#" + row.v : "–"));
        tr.appendChild(el("td", "num", row.s ? "#" + row.s : "–"));
        tr.appendChild(el("td", "num", row.formula));
        grid.tbody.appendChild(tr);
      });

      if (!scored.length) { note.textContent = "With both arms off there is nothing to fuse."; return; }
      var winner = scored[0].item;
      var answerAt = scored.findIndex(function (row) { return row.item.answer; }) + 1;
      var words = winner.answer
        ? "A, the passage that answers the question, comes first."
        : winner.id + " comes first, but it does not answer the question; A is #" + answerAt + ".";
      if (useVector.checked !== useKeyword.checked) {
        words += " With one arm only, fusion just repeats that arm's order.";
      } else if (tip !== null && k < tip) {
        words += " With a small k, a single first place outweighs agreement between both lists. Try k = " + tip + " or more.";
      } else if (tip !== null) {
        words += " A passage that both arms rank fairly well beats one that a single arm ranks first. Below k = " + tip + ", B's single first place would win instead.";
      }
      note.textContent = words;
    }
    [kInput, useVector, useKeyword].forEach(function (input) {
      input.addEventListener("input", render);
      input.addEventListener("change", render);
    });
    render();
  }

  // --------------------------------------------------------------- glossary ---
  var CONCEPTS = [
    {
      term: "Chunking with overlap",
      short: "Documents are cut into passages of about 400 tokens along headings and paragraphs.",
      body: "Search returns passages, not whole documents, and a model can only read a few. Cutting along headings and paragraphs keeps each chunk about one thing. The last 60 or so tokens of a chunk are repeated at the start of the next, so a sentence that straddles a cut is still whole on one side. Each chunk also records where every page begins inside it (its page map), which is how a citation names the exact page its quote starts on.",
      code: ["app/ingestion/chunker.py", "app/ingestion/pdf_parser.py"]
    },
    {
      term: "Contextual retrieval",
      short: "Each chunk is stored with a one-line preamble saying where it comes from.",
      body: "Cut out of its document, “the limit is 10 ppm” no longer says which product or procedure it is about. Anthropic's Contextual Retrieval technique prepends a short situating preamble to every chunk before it is embedded and indexed, which helps both search arms. This free demo builds the preamble from structure (document title, section heading, pages) at no cost; with a Claude API key the engine asks Claude Haiku to write it instead.",
      code: ["app/ingestion/contextualizer.py"]
    },
    {
      term: "Embedding",
      short: "A list of numbers that captures what a text means.",
      body: "An embedding model maps a text to a vector: here 384 numbers from BAAI/bge-small-en-v1.5, which runs on the server itself through ONNX Runtime, with no API call. Texts with similar meaning get nearby vectors, so “keep records” finds “retain records”. Embeddings are weak on exact identifiers such as SOP-QA-017, which is why keyword search runs as well.",
      code: ["app/retrieval/embedder.py"]
    },
    {
      term: "Cosine distance",
      short: "How differently two vectors point, ignoring their length.",
      body: "Cosine distance is 0 for vectors pointing the same way, 1 for unrelated ones and 2 for opposite ones. pgvector's <=> operator computes it, and the vector arm returns the chunks with the smallest distance to the question.",
      code: ["app/retrieval/hybrid_search.py"]
    },
    {
      term: "HNSW index",
      short: "A graph that finds near neighbours without comparing the question to every chunk.",
      body: "Comparing the question with every stored vector is exact but slow at scale. HNSW (Hierarchical Navigable Small World) links each vector to a few neighbours in layered graphs and walks toward the question. m = 16 links per node and ef_construction = 64 set how carefully the graph is built; ef_search sets how many candidates a query keeps while walking. It is approximate: a lot of speed for a small chance of missing a neighbour.",
      code: ["app/database.py"]
    },
    {
      term: "Full-text search: tsvector, tsquery, GIN",
      short: "PostgreSQL's built-in keyword search.",
      body: "Each chunk's text is stored as a tsvector, a list of normalised words in which plural and verb endings are stripped, so “records” and “record” match. A GIN index maps every word to the chunks that contain it, like the index at the back of a book. The question becomes a tsquery whose words are joined with OR, so a chunk matching some of them still ranks, and ts_rank_cd scores how densely they occur.",
      code: ["app/retrieval/hybrid_search.py", "app/database.py"]
    },
    {
      term: "Reciprocal Rank Fusion (RRF)",
      short: "Merges two ranked lists using positions, not scores.",
      body: "A chunk earns 1 / (k + rank) from each list that contains it, and the sums decide the fused order. Because only ranks are used, the two arms' incompatible scores never need calibrating. k = 60, the value from the original paper, damps the advantage of a first place so agreement between the lists counts. Here it runs inside the same SQL statement as both searches.",
      code: ["app/retrieval/hybrid_search.py"]
    },
    {
      term: "Cross-encoder re-ranking",
      short: "A slower, sharper model that reads question and passage together.",
      body: "The embedding model is a bi-encoder: it encodes question and passage separately, so the comparison is only as good as two summaries. A cross-encoder (here ms-marco-MiniLM-L-6-v2) reads both in one input and scores the pair, catching details such as “critical” versus “major”. It is too slow for every chunk, so it scores the fused candidates only and keeps the best 3.",
      code: ["app/retrieval/reranker.py"]
    },
    {
      term: "Structured output",
      short: "The model must reply with JSON that matches a schema.",
      body: "The request carries a JSON schema: answer, is_grounded, confidence_score, and citations made of a chunk id and a verbatim quote. Models that support strict mode are constrained to it while they write; for free models that refuse it, the engine retries in plain JSON mode with the schema spelled out, then validates the reply with Pydantic. A reply that does not parse is an error, never an answer.",
      code: ["app/generator/grounded_llm.py", "app/openrouter.py"]
    },
    {
      term: "Quote verification",
      short: "Every quote is looked up in the stored chunk; the ones that are not there are dropped.",
      body: "Models paraphrase and misremember numbers even when told not to. After generation the server searches for each quote in the chunk it names, tolerating only differences of case, spacing and typography, drops what it cannot find, and withholds the answer when nothing survives. The citation shows the stored text, and its title and page come from the database.",
      code: ["app/generator/grounded_llm.py"]
    },
    {
      term: "Saying “not in the documents”",
      short: "An explicit negative answer is a valid result.",
      body: "When retrieval finds nothing relevant, or the model reports that the passages do not answer the question, the response carries the status insufficient_context and no citations. In compliance work a confident wrong answer costs more than an honest gap.",
      code: ["app/generator/grounded_llm.py", "app/schemas.py"]
    },
    {
      term: "Prompt-injection defence",
      short: "Document text is treated as data, never as instructions.",
      body: "A document could contain a line such as “ignore your instructions”. Passages are wrapped in tags with a random suffix per request, so a document cannot close its own tag or fake a question. The rules tell the model to quote such text, never follow it, and quote verification limits what a hijacked reply could still claim.",
      code: ["app/generator/grounded_llm.py"]
    },
    {
      term: "A public demo at zero cost",
      short: "Free models with fallback, local models, and a request budget.",
      body: "Generation uses free OpenRouter models tried in order, falling back when one is rate limited or withdrawn. Embedding and re-ranking run locally. A budget caps questions per visitor and for everyone, per minute and per day, and checks OpenRouter's own count of today's free requests before each question.",
      code: ["app/openrouter.py", "app/budget.py"]
    }
  ];

  function concepts(container) {
    container.textContent = "";
    var list = el("div", "xp-concepts");
    CONCEPTS.forEach(function (c) {
      var d = el("details", "xp-concept");
      var s = el("summary");
      s.appendChild(el("span", "xp-name", c.term));
      s.appendChild(el("span", "xp-sum", c.short));
      d.appendChild(s);
      var body = el("div", "xp-body");
      body.appendChild(el("p", null, c.body));
      body.appendChild(codeLinks(c.code));
      d.appendChild(body);
      list.appendChild(d);
    });
    container.appendChild(list);
  }

  window.RagExplain = { walkthrough: walkthrough, sandbox: sandbox, concepts: concepts };
})();
