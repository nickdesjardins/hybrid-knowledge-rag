// Teaching layer shared by the project page (site/) and the playground (app/playground/static/).
//
//   RagExplain.walkthrough(node, response, context)  every stage of one answer, with its numbers,
//                                                    and the vector map: context.vectorMap when the
//                                                    page passes one, else response.vector_map
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
  // Whether the server replaced the model's draft (pipeline.draft_withheld): true or false when it
  // says so, null for a response recorded before the field existed.
  function draftWithheld(p) { return p.draft_withheld === true ? true : p.draft_withheld === false ? false : null; }

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
      { key: "rerank", label: p.reranker_model === "none" ? "Keep top " + p.passages_to_model : "Re-rank", value: t.rerank_ms || 0, text: ms(t.rerank_ms) }
    ];
    if (t.generate_ms !== null && t.generate_ms !== undefined) {
      timeParts.push({ key: "generate", label: "Generate + verify", value: t.generate_ms, text: ms(t.generate_ms) });
    }
    container.appendChild(stackedBar(timeParts,
      "Where the time went. Retrieval runs on the server itself in milliseconds; the language model is a call over the network to a shared free model, so it usually dominates."));

    // A map the page passes was replayed later by scripts/record_runs.py --maps-only, so it is newer
    // than one inside the recorded response; the live playground passes none and uses the answer's.
    var map = context.vectorMap || r.vector_map;
    var noRerank = p.reranker_model === "none";
    var hasMap = !!(map && map.points && map.points.length && map.query);
    var shownMap = { numbers: false, rankings: false };
    if (hasMap) {
      try {
        shownMap = vectorMap(container, r, map);
      } catch (e) {
        // A teaching aid must never take the walkthrough down with it.
        hasMap = false;
        shownMap = { numbers: false, rankings: false };
        var broken = container.querySelector(".xp-vmap");
        if (broken) container.removeChild(broken);
        if (window.console) window.console.warn("vector map not shown:", e);
      }
    }

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
    if (shownMap.numbers) {
      s1.body.appendChild(el("p", "xp-note",
        "“The numbers behind the dots” above shows all " + p.embedding_dimensions + " of them, next to the numbers of a chunk they were compared with."));
    }
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
    if (hasMap) {
      s2.body.appendChild(el("p", "xp-note",
        "The map above shows every chunk; its “Distance from the question” view shows exactly how far each one is from the question. Number the dots by Vector, then by Keyword, to see what each arm picked."));
    }
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
    // RERANK_PROVIDER=none keeps the fused order: no cross-encoder ran, so none may be described.
    var s4 = step(4, noRerank ? "Keep the top fused candidates" : "Re-rank with a cross-encoder",
      "kept " + p.passages_to_model + " of " + plural(p.candidates, "candidate"), t.rerank_ms);
    s4.body.appendChild(para(null, [noRerank
      ? "No re-ranker is configured ([[RERANK_PROVIDER=none]]), so the first " + plural(p.passages_to_model, "candidate") + " in fused order went on to the model. " +
        "Each one's score is its RRF score divided by the top candidate's."
      : "The cross-encoder [[" + p.reranker_model + "]] read the question and each fused candidate together, as one input, and scored how well the passage answers it. " +
        "The best " + p.passages_to_model + " went on to the model."
    ]));
    if (passages.length) {
      s4.body.appendChild(table(
        ["Passage", "Fused position", noRerank ? "Position kept" : "Re-ranked position", noRerank ? "RRF score, scaled" : "Cross-encoder score", "Cited"],
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
      if (!noRerank) {
        s4.body.appendChild(el("p", "xp-note", lifted
          ? "The re-ranker lifted a passage from fused position #" + lifted.fused_rank + " to #" + lifted.rerank_position + ": it noticed something rank fusion could not."
          : "Here the re-ranker agreed with the fused order."));
      } else if (!lifted) {
        s4.body.appendChild(el("p", "xp-note", "No re-ranker ran, so the fused order was kept."));
      }
    }
    if (shownMap.rankings) {
      s4.body.appendChild(el("p", "xp-note",
        "“How the four rankings disagree” above follows every candidate from its vector and keyword ranks, through fusion, to " +
        (noRerank ? "the passages kept (no re-ranker ran)." : "the cross-encoder's order.")));
    }
    s4.body.appendChild(why(noRerank
      ? "a cross-encoder reads question and passage side by side, which is sharper than comparing two embeddings, but this deployment switched it off, so the fused order alone decides what the model reads."
      : "the embedding model reads question and passage separately and compares two summaries. A cross-encoder reads them side by side, which is slower but much sharper, so it only runs on the short list. Its scores are raw logits: higher is better, and negative values are normal."));
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
    // The server says whether it replaced the draft (the model claimed grounding, no quote
    // matched). Older recordings lack the flag: then only what is true either way is said.
    var withheld = draftWithheld(p);
    var verdict;
    if (r.status === "answered") verdict = verified + " of " + plural(proposed, "quote") + " matched the document";
    else if (r.status === "refused") verdict = "the model declined to answer";
    else if (withheld === true) verdict = proposed ? "no quote matched, so the draft was withheld" : "no quote given, so the draft was withheld";
    else if (proposed && withheld === null) verdict = "no quote was kept as a citation";
    else if (proposed) verdict = "the model said the passages fall short";
    else verdict = "the documents do not answer this";
    var s6 = step(6, "Verify every quote", verdict, null);
    if (r.status === "answered") {
      s6.body.appendChild(para(null, [
        "The model proposed " + plural(proposed, "quote") + ". The server searched for each one in the stored text of the passage it names, ignoring only differences of case, spacing and typography, and kept " + verified + ". " +
        "Each citation's page comes from that chunk's page map: the page on which the quote begins. The model never supplies a page number."
      ]));
    } else if (r.status !== "refused" && withheld === true) {
      s6.body.appendChild(para(null, [proposed
        ? "The model claimed a grounded answer and proposed " + plural(proposed, "quote") + ", but " + (proposed === 1 ? "it" : "none of them") + " could " + (proposed === 1 ? "not " : "") + "be found in the stored passages. An answer without a verifiable quote is not shown."
        : "The model claimed a grounded answer but gave no quote to check it against. An answer without a verifiable quote is not shown."
      ]));
    } else if (r.status !== "refused" && proposed && withheld === false) {
      s6.body.appendChild(para(null, [
        "The model proposed " + plural(proposed, "quote") + " but itself reported that these passages do not answer the question, so the engine returns [[" + r.status + "]] with no citations."
      ]));
    } else if (r.status !== "refused" && proposed) {
      s6.body.appendChild(para(null, [
        "The model proposed " + plural(proposed, "quote") + " and the server " + (proposed === 1 ? "did not keep it as a citation: either it could not" : "kept none of them as citations: either none could") +
        " be found in the stored passages, or the model itself reported that the passages do not answer the question. " +
        "The engine returns [[" + r.status + "]] with no citations."
      ]));
    } else if (r.status === "refused") {
      s6.body.appendChild(para(null, [
        "The model declined to answer, so there was nothing to verify. The engine returns the status [[" + r.status + "]] with no citations."
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

  // ------------------------------------------------------------- vector map ---
  // "Where your question landed": every chunk of the corpus placed by its embedding with the
  // question among them, then the four rankings side by side, the numbers behind one similarity,
  // and a table of it all. The server sends the coordinates, similarities and ranks (vector_map,
  // built by app/retrieval/vector_map.py); every sentence here is computed from them.
  var SVGNS = "http://www.w3.org/2000/svg";
  var LENSES = [
    { prop: "dense", label: "Vector", short: "Vector", sub: "cosine", arm: "vector search" },
    { prop: "sparse", label: "Keyword", short: "Keyword", sub: "full text", arm: "keyword search" },
    { prop: "fused", label: "Fused (RRF)", short: "Fused", sub: "RRF", arm: "rank fusion" },
    { prop: "rerank", label: "Re-ranked", short: "Re-ranked", sub: "cross-encoder", arm: "the cross-encoder" }
  ];
  // With RERANK_PROVIDER=none the last column is the fused order cut to the passages sent on.
  var KEPT_LENS = { prop: "rerank", label: "Kept (RRF order)", short: "Kept", sub: "no re-ranker", arm: "rank fusion, kept without a re-ranker" };
  function lensesFor(m) { return m.noRerank ? LENSES.slice(0, 3).concat([KEPT_LENS]) : LENSES; }
  // Words for an answer that quoted nothing. Similarity scales differ between embedding models
  // (text-embedding-3-small often scores a relevant passage around 0.4 to 0.6), so "nothing is
  // close" is only said for a model it was measured on. BAAI/bge-small-en-v1.5 on the sample
  // corpus: every answerable recorded question has a chunk at 0.72 or more, the unanswerable 0.52.
  var FAR_SIMILARITY = { "BAAI/bge-small-en-v1.5": 0.6 };
  // Above this many dots the distance view's disc is too crowded to clear, and pulling dots apart
  // costs O(n²) per pass: they are drawn where they fall, at their exact distance.
  var RELAX_MAX_DOTS = 200;
  // How far apart the distance view keeps two dots when one wears a sent or quoted ring (twice
  // the ring's radius plus its stroke), and two plain dots.
  var RING_GAP = 22;
  var DOT_GAP = 16;
  var GRID_COLUMNS = 24;
  var NUMBER_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
  var mapCount = 0;
  var liveMaps = [];
  var resizeBound = false;
  var resizeTimer = null;

  function svg(tag, attrs, cls) {
    var node = document.createElementNS(SVGNS, tag);
    if (cls) node.setAttribute("class", cls);
    Object.keys(attrs || {}).forEach(function (name) { node.setAttribute(name, String(attrs[name])); });
    return node;
  }
  function svgText(text, attrs, cls) {
    var node = svg("text", attrs, cls);
    node.textContent = text;
    return node;
  }
  // Fixed decimals with a true minus sign, and never "−0.000".
  function num(v, digits) {
    var text = Math.abs(v).toFixed(digits);
    return (v < 0 && Number(text) !== 0 ? "−" : "") + text;
  }
  function count(n, one, many) { return (n < NUMBER_WORDS.length ? NUMBER_WORDS[n] : String(n)) + " " + (n === 1 ? one : many); }
  function andList(parts) { return parts.length < 2 ? parts.join("") : parts.slice(0, -1).join(", ") + " and " + parts[parts.length - 1]; }
  function rankText(v) { return v ? "#" + v : "–"; }
  function overlaps(a, b) { return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function angleBetween(a, b) { return Math.atan2(Math.sin(a - b), Math.cos(a - b)); }
  // Positions go through CSSOM (allowed by the pages' CSP) so a view change can animate them.
  function moveTo(node, p) { node.style.transform = "translate(" + p.x.toFixed(1) + "px, " + p.y.toFixed(1) + "px)"; }
  function reducedMotion() { return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches); }

  // "SOP-QA-017 Cleaning Validation …" → "SOP-QA-017"; a title with no ID is shortened instead.
  function docCode(title) {
    var first = title.split(/\s+/)[0];
    return /^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$/.test(first) && first !== title ? first : shorten(title, 22);
  }
  // A revision at the end of a title: "(Rev 4)", "(v2)", or a bare "Rev 4" / "v2".
  var REV_PAREN = /\s*\(((?:rev(?:ision)?|v)[\s.\d][^)]*)\)\s*$/i;
  var REV_BARE = /\s+((?:rev(?:ision)?|v)\.?\s*\d[\w.]*)\s*$/i;
  function docName(doc) {
    var base = doc.base || doc.code;
    var title = doc.title.replace(REV_PAREN, "");
    if (doc.tagKind === "rev") title = title.replace(REV_BARE, "");
    var name = base === doc.title || doc.title.indexOf(base + " ") !== 0
      ? shorten(doc.tagKind === "rev" ? title : doc.title, 56)
      : base + " · " + shorten(title.slice(base.length).trim(), 56);
    if (doc.tagKind === "rev") return name + " (" + doc.tag + ")";
    return doc.tag ? name + " " + doc.tag : name;
  }
  // Two documents with one code (two revisions of a SOP, or one PDF ingested twice) would share a
  // map label, a legend line and chunk names: the revision, or else an ordinal, tells them apart.
  // Documents that do not collide keep their code as it is.
  function disambiguateDocs(docs) {
    var groups = {};
    docs.forEach(function (doc) {
      doc.base = doc.code;
      if (!Object.prototype.hasOwnProperty.call(groups, doc.code)) groups[doc.code] = [];
      groups[doc.code].push(doc);
    });
    Object.keys(groups).forEach(function (code) {
      var group = groups[code];
      if (group.length < 2) return;
      var tags = group.map(function (doc) {
        var found = REV_PAREN.exec(doc.title) || REV_BARE.exec(doc.title);
        return found ? found[1].replace(/\s+/g, " ") : "";
      });
      var distinct = tags.every(function (tag, k) { return tag && tags.indexOf(tag) === k; });
      group.forEach(function (doc, k) {
        // "(2)", not "#2": "#" means a rank everywhere else on this page.
        doc.tag = distinct ? tags[k] : "(" + (k + 1) + ")";
        doc.tagKind = distinct ? "rev" : "ordinal";
        doc.code = doc.base + " " + doc.tag;
      });
    });
  }
  function chunkName(d) { return d.doc.code + " chunk " + d.chunk; }
  // A chunk sent to the model wears a ring (dashed, or solid when quoted) of radius 10.
  function ringed(d) { return !!(d.cited || d.rerank); }
  // Where a rank badge is tried around its dot, in order: the four corners from the upper right,
  // then the sides (screen angles: y grows downwards).
  var BADGE_DIRS = [-45, -135, 45, 135, 0, 180, -90, 90].map(function (deg) { return deg * Math.PI / 180; });
  // How many chunks have a chunk of their own document as their nearest neighbour on the map:
  // whether documents form regions is read from this map, not assumed. (All pairs: skipped for a
  // very large corpus.)
  function sameDocNeighbours(m) {
    if (m.docs.length < 2 || m.points.length < 3 || m.points.length > 2000) return null;
    var same = 0;
    m.points.forEach(function (d) {
      var best = null, bestDist = Infinity;
      m.points.forEach(function (o) {
        if (o === d) return;
        var dist = Math.pow(o.x - d.x, 2) + Math.pow(o.y - d.y, 2);
        if (dist < bestDist) { bestDist = dist; best = o; }
      });
      if (best && best.doc === d.doc) same += 1;
    });
    return { same: same, total: m.points.length };
  }

  // Equal-area marks, one per document slot: colour is never the only clue to a document.
  function shapePath(slot, r) {
    function f(v) { return String(Math.round(v * 100) / 100); }
    var pts = [];
    var i, s;
    if (slot === 1) { s = r * 0.886; pts = [[-s, -s], [s, -s], [s, s], [-s, s]]; }
    else if (slot === 2) { s = r * 1.25; pts = [[0, -s], [s, 0], [0, s], [-s, 0]]; }
    else if (slot === 3) { pts = [[0, -r * 1.555], [r * 1.345, r * 0.777], [-r * 1.345, r * 0.777]]; }
    else if (slot === 4) {
      for (i = 0; i < 6; i++) pts.push([r * 1.1 * Math.cos(Math.PI / 6 + i * Math.PI / 3), r * 1.1 * Math.sin(Math.PI / 6 + i * Math.PI / 3)]);
    } else {
      return "M" + f(-r) + " 0A" + f(r) + " " + f(r) + " 0 1 0 " + f(r) + " 0A" + f(r) + " " + f(r) + " 0 1 0 " + f(-r) + " 0Z";
    }
    return "M" + pts.map(function (p) { return f(p[0]) + " " + f(p[1]); }).join("L") + "Z";
  }
  function starPath(outer) {
    var pts = [];
    for (var i = 0; i < 10; i++) {
      var rad = i % 2 ? outer * 0.45 : outer;
      var a = -Math.PI / 2 + i * Math.PI / 5;
      pts.push((Math.round(rad * Math.cos(a) * 100) / 100) + " " + (Math.round(rad * Math.sin(a) * 100) / 100));
    }
    return "M" + pts.join("L") + "Z";
  }
  // Small keys for legends, the detail card, chips and table rows (decorative: text says it all).
  function keyIcon(cls, parts) {
    var box = svg("svg", { viewBox: "-9 -9 18 18", width: 16, height: 16, "aria-hidden": "true", focusable: "false" }, "xp-vm-key" + (cls ? " " + cls : ""));
    parts.forEach(function (part) { box.appendChild(part); });
    return box;
  }
  function docKey(slot) {
    return keyIcon("xp-vm-doc" + (slot + 1), [svg("path", { d: shapePath(slot, 4.4) }, "xp-vm-shape xp-vm-shape--flat")]);
  }

  function mapModel(r, map) {
    var cited = {};
    (r.citations || []).forEach(function (c) { cited[c.chunk_id] = true; });
    var scores = {};
    (r.retrieved_passages || []).forEach(function (q) { scores[q.chunk_id] = q.relevance_score; });
    var vectors = map.vectors || {};
    var docs = [];
    var docById = {};
    var points = map.points.map(function (pt, i) {
      var doc = docById[pt.document_id];
      if (!doc) {
        var title = String(pt.document_title || "Untitled document");
        // Documents 5 and later share one grey slot: colours are never cycled.
        doc = { id: pt.document_id, title: title, code: docCode(title), slot: Math.min(docs.length, 4), chunks: 0 };
        docById[pt.document_id] = doc;
        docs.push(doc);
      }
      doc.chunks += 1;
      return {
        id: pt.chunk_id, i: i, doc: doc, x: Number(pt.x), y: Number(pt.y), sim: Number(pt.similarity),
        page: pt.page_number, chunk: Number(pt.chunk_index) + 1, preview: String(pt.preview || ""),
        searched: pt.searched !== false,
        dense: pt.dense_rank || null, sparse: pt.sparse_rank || null,
        fused: pt.fused_rank || null, rerank: pt.rerank_position || null,
        cited: !!cited[pt.chunk_id],
        score: scores[pt.chunk_id] === undefined ? null : scores[pt.chunk_id],
        vector: vectors[pt.chunk_id] || null
      };
    });
    disambiguateDocs(docs);
    var byId = {};
    points.forEach(function (d) { byId[d.id] = d; });
    var p = r.pipeline || {};
    // A chunk quoted twice is one passage.
    var seenQuoted = {};
    return {
      points: points, docs: docs, byId: byId,
      bySim: points.slice().sort(function (a, b) { return b.sim - a.sim || a.i - b.i; }),
      quoted: (r.citations || []).map(function (c) { return byId[c.chunk_id]; }).filter(function (d) {
        if (!d || seenQuoted[d.id]) return false;
        seenQuoted[d.id] = true;
        return true;
      }),
      first: function (prop) { return points.filter(function (d) { return d[prop] === 1; })[0] || null; },
      dims: map.dimensions || (map.query.vector || []).length,
      variance: map.explained_variance || [],
      query: map.query,
      toModel: p.passages_to_model || 0,
      rrfK: p.rrf_k,
      embeddingModel: String(p.embedding_model || ""),
      noRerank: p.reranker_model === "none",
      // Each search arm and the fused list keep at most candidate_limit chunks; when the fused
      // list is that long, rank fusion may have cut chunks that one arm did return.
      candidateLimit: p.candidate_limit || 0,
      truncated: !!(p.candidate_limit && p.candidates >= p.candidate_limit),
      quotesProposed: p.quotes_proposed || 0,
      withheld: draftWithheld(p),
      status: r.status,
      sent: r.retrieved_passages || []
    };
  }

  // The sentence under the map: what this answer teaches, computed from its own numbers. With the
  // playground's document filter only part of the corpus was searched; the words then say so.
  function mapLesson(m, view) {
    var c = m.quoted[0];
    var searched = m.points.filter(function (d) { return d.searched; });
    var left = m.points.length - searched.length;
    var words;
    var subject = !c ? "" : m.quoted.length > 1
      ? "The first of the " + count(m.quoted.length, "passage", "passages") + " the answer quotes, " + chunkName(c) + ","
      : "The passage the answer quotes, " + chunkName(c) + ",";
    if (c && c.dense === 1) {
      words = subject + " was also the vector search's first pick (similarity " + num(c.sim, 2) + "). " +
        "Keyword search " + (c.sparse ? "ranked it #" + c.sparse : "did not return it") +
        ", rank fusion put it #" + c.fused +
        (!c.rerank ? "" : m.noRerank ? " and it went to the model as passage #" + c.rerank + " (no re-ranker ran)" : " and the cross-encoder #" + c.rerank) + ".";
    } else if (c) {
      var closer = searched.filter(function (d) { return d.sim > c.sim; }).length;
      var closerOut = m.points.filter(function (d) { return !d.searched && d.sim > c.sim; }).length;
      words = subject + " was " + andList([
        c.dense
          ? "#" + c.dense + " by vector search (similarity " + num(c.sim, 2) + ": " +
            count(closer, left ? "searched chunk" : "chunk", left ? "searched chunks" : "chunks") + (closer === 1 ? " was" : " were") + " closer to the question" +
            (closerOut ? ", and the filter left out " + count(closerOut, "closer one", "closer ones") : "") + ")"
          : "not returned by vector search (similarity " + num(c.sim, 2) + ")",
        c.sparse ? "#" + c.sparse + " by keyword search" : "not returned by keyword search",
        c.fused ? "#" + c.fused + " after rank fusion" : "not a fused candidate",
        c.rerank ? "#" + c.rerank + (m.noRerank ? " of the passages kept" : " after re-ranking") : (m.noRerank ? "not kept" : "not re-ranked")
      ]) + ".";
      if (m.toModel && (!c.dense || c.dense > m.toModel)) {
        words += " Vector search alone, keeping its top " + m.toModel + ", would have missed it.";
      }
    } else if (m.status === "answered") {
      // Quoted, but the quoted chunk is not on this map (it changed in between): say nothing.
      words = "";
    } else {
      words = unquotedLesson(m, searched, left);
    }
    if (view === "map") {
      var q = m.query;
      var onMap = m.points.reduce(function (a, b) {
        return Math.pow(b.x - q.x, 2) + Math.pow(b.y - q.y, 2) < Math.pow(a.x - q.x, 2) + Math.pow(a.y - q.y, 2) ? b : a;
      });
      var best = m.bySim[0];
      if (onMap !== best && best.sim > onMap.sim) {
        // Enough decimals that the two numbers differ on screen.
        var digits = 2;
        while (digits < 4 && num(best.sim, digits) === num(onMap.sim, digits)) digits += 1;
        words += " On this flat map the question looks closest to " + chunkName(onMap) + (onMap.searched ? "," : ", which the filter left out,") +
          " but in all " + m.dims + " dimensions its nearest chunk is " + chunkName(best) +
          " (similarity " + num(best.sim, digits) + " against " + num(onMap.sim, digits) + (best.searched ? "" : "; the filter left it out") +
          "); switch to the distance view to see each chunk's exact distance from the question.";
      }
    }
    return words.trim();
  }

  // An answer that quoted nothing: how near the nearest searched chunk was, and what happened next.
  function unquotedLesson(m, searched, left) {
    var words;
    var bestAll = m.bySim[0];
    var cutoff = Object.prototype.hasOwnProperty.call(FAR_SIMILARITY, m.embeddingModel) ? FAR_SIMILARITY[m.embeddingModel] : null;
    if (!searched.length) {
      // No candidates, so no model call either (the engine answers that without one).
      return "The document filter left no chunk of this corpus to search, so nothing was retrieved.";
    } else {
      var nearest = searched.reduce(function (a, b) { return b.sim > a.sim ? b : a; });
      // A proposed quote, or a draft the model claimed was grounded, means the model saw something
      // in these chunks: no "far" verdict then.
      var far = cutoff !== null && nearest.sim < cutoff && !m.quotesProposed && m.withheld !== true;
      if (!left) {
        words = far
          ? "Nothing in the corpus is close to this question: the nearest chunk has a similarity of " + num(nearest.sim, 2) + "."
          : "No passage was quoted. The nearest chunk has a similarity of " + num(nearest.sim, 2) +
            (cutoff === null ? "; what counts as close depends on the embedding model" : "") + ".";
      } else {
        var codes = [];
        searched.forEach(function (d) { if (codes.indexOf(d.doc.code) < 0) codes.push(d.doc.code); });
        var scope = codes.length === 1 ? codes[0] + ", the only document searched," : "the " + count(codes.length, "document", "documents") + " searched";
        words = far
          ? "Nothing in " + scope + " is close to this question: its nearest chunk has a similarity of " + num(nearest.sim, 2) + "."
          : "No passage was quoted. The nearest chunk in " + scope + " has a similarity of " + num(nearest.sim, 2) + ".";
        if (!bestAll.searched && bestAll.sim > nearest.sim) {
          words += " The document filter left out " + count(left, "chunk", "chunks") + "; the closest of them, " + chunkName(bestAll) +
            ", has a similarity of " + num(bestAll.sim, 2) + ".";
        }
      }
    }
    // Whether the draft was withheld comes from the server; without that flag (older recordings)
    // only what is true either way is said.
    var ending = m.status === "refused" ? "the model declined to answer."
      : m.withheld === true
        ? (m.quotesProposed === 1 ? "the model's quote could not be found in them, so its draft was withheld."
          : m.quotesProposed ? "none of the model's quotes could be found in them, so its draft was withheld."
          : "the model claimed an answer but quoted nothing, so its draft was withheld.")
      : m.quotesProposed && m.withheld === null ? (m.quotesProposed === 1 ? "the server did not keep the model's quote as a citation." : "the server kept none of the model's quotes as citations.")
      : m.quotesProposed ? "the model reported that they do not fully answer it, so " + (m.quotesProposed === 1 ? "its quote is not shown as a citation." : "none of its quotes is shown as a citation.")
      : m.status === "insufficient_context" ? "the model reported that they do not answer it."
      : "";
    if (!m.points.some(function (d) { return d.fused; })) {
      words += " Neither search arm returned a chunk for it.";
    } else {
      // Only a cross-encoder scores below zero; RERANK_PROVIDER=none passes scaled RRF scores on.
      var negative = !m.noRerank && m.sent.length && m.sent.every(function (q) { return q.relevance_score < 0; });
      words += " Search still returns its nearest chunks for any question" +
        (negative ? "; the cross-encoder scored all " + count(m.sent.length, "passage", "passages") + " it kept below zero" + (ending ? ", and " + ending : ".")
          : ending ? "; " + ending : ".");
    }
    return words;
  }

  // Map view: the two PCA axes on one scale, so a distance on screen means the same either way.
  var MAP_PAD = { l: 38, r: 28, t: 22, b: 38 };
  function mapExtent(m) {
    var xs = m.points.map(function (d) { return d.x; }).concat([m.query.x]);
    var ys = m.points.map(function (d) { return d.y; }).concat([m.query.y]);
    return { x0: Math.min.apply(null, xs), x1: Math.max.apply(null, xs), y0: Math.min.apply(null, ys), y1: Math.max.apply(null, ys) };
  }
  // The plot takes the shape of the data (within 280 to 520 px), so a square cloud is not squeezed
  // into a wide strip with empty space on both sides.
  function mapHeight(m, w) {
    var e = mapExtent(m);
    var aspect = Math.max((e.x1 - e.x0) / Math.max(e.y1 - e.y0, 1e-9), 0.5);
    return Math.round(clamp((w - MAP_PAD.l - MAP_PAD.r) / aspect + MAP_PAD.t + MAP_PAD.b, 280, 520));
  }
  function mapLayout(m, w, h) {
    var pad = MAP_PAD;
    var e = mapExtent(m);
    var x0 = e.x0, x1 = e.x1, y0 = e.y0, y1 = e.y1;
    var iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    var s = Math.min(iw / Math.max(x1 - x0, 1e-9), ih / Math.max(y1 - y0, 1e-9));
    var ox = pad.l + (iw - (x1 - x0) * s) / 2;
    var oy = pad.t + (ih - (y1 - y0) * s) / 2;
    function at(x, y) { return { x: ox + (x - x0) * s, y: oy + (y1 - y) * s }; }
    return {
      pos: m.points.map(function (d) { return at(d.x, d.y); }),
      star: at(m.query.x, m.query.y),
      origin: at(0, 0),
      box: { x0: pad.l, y0: pad.t, x1: w - pad.r, y1: h - pad.b },
      bounds: { x0: 24, y0: 2, x1: w - 2, y1: h - 22 },
      reserved: []
    };
  }

  // Overlapping dots at almost the same distance are pulled apart along their ring, a little,
  // the same way every time; a dot wearing a sent or quoted ring is kept a ring's width from the
  // others. Only the angle moves, never the distance. Maps of more than RELAX_MAX_DOTS dots are
  // left as they fall.
  function relax(nodes) {
    if (nodes.length > RELAX_MAX_DOTS) return;
    for (var pass = 0; pass < 60; pass++) {
      var moved = false;
      for (var i = 0; i < nodes.length; i++) {
        for (var j = i + 1; j < nodes.length; j++) {
          var a = nodes[i], b = nodes[j];
          var gap = a.ring || b.ring ? RING_GAP : DOT_GAP;
          // Two dots are never closer than the difference of their distances: skip the trigonometry.
          if (Math.abs(a.rad - b.rad) >= gap) continue;
          var dx = a.rad * Math.cos(a.a) - b.rad * Math.cos(b.a);
          var dy = a.rad * Math.sin(a.a) - b.rad * Math.sin(b.a);
          var dist = Math.sqrt(dx * dx + dy * dy);
          if (dist >= gap) continue;
          var dir = angleBetween(b.a, a.a) >= 0 ? 1 : -1;
          var push = Math.min(0.04, (gap - dist) / Math.max(20, (a.rad + b.rad) / 2) / 2);
          b.a += dir * push;
          a.a -= dir * push;
          moved = true;
        }
      }
      nodes.forEach(function (n) { n.a = n.a0 + clamp(angleBetween(n.a, n.a0), -0.5, 0.5); });
      if (!moved) break;
    }
  }

  // Distance view: the question at the centre, each chunk at its exact cosine distance (1 - its
  // similarity). The angle only places the dot: it is the chunk's angle around the map's origin
  // (the average chunk vector, where the map's axes cross), not a direction from the question.
  function distanceLayout(m, w, h) {
    var cx = w / 2, cy = h / 2;
    var R = Math.max(60, Math.min(w, h) / 2 - 26);
    var far = m.points.reduce(function (most, d) { return Math.max(most, 1 - d.sim); }, 0);
    var domain = Math.max(0.3, Math.ceil(far * 10 - 1e-9) / 10);
    var nodes = m.points.map(function (d) {
      var a = Math.atan2(d.y, d.x);
      return { a0: a, a: a, rad: R * (1 - d.sim) / domain, ring: !!(d.cited || d.rerank) };
    });
    relax(nodes);
    var pos = nodes.map(function (node) { return { x: cx + node.rad * Math.cos(node.a), y: cy - node.rad * Math.sin(node.a) }; });
    var zones = pos.map(function (p) { return { x0: p.x - 11, x1: p.x + 19, y0: p.y - 18, y1: p.y + 11 }; });
    var rings = Math.round(domain * 10);
    var spacing = R / rings;
    var every = Math.max(1, Math.ceil(13 / spacing));
    function ringBoxes(a) {
      var boxes = [];
      for (var t = every; t <= rings; t += every) {
        var lx = cx + spacing * t * Math.cos(a), ly = cy - spacing * t * Math.sin(a);
        boxes.push({ x0: lx - 12, x1: lx + 12, y0: ly - 8, y1: ly + 6 });
      }
      return boxes;
    }
    // Ring labels go where they touch no dot or rank badge, preferring the upper right.
    var angle = 0, bestCost = Infinity;
    for (var k = 0; k < 24; k++) {
      var a = k * Math.PI / 12;
      var cost = 0.01 * Math.abs(angleBetween(a, Math.PI / 3));
      ringBoxes(a).forEach(function (box) {
        zones.forEach(function (zone) { if (overlaps(box, zone)) cost += 1; });
      });
      if (cost < bestCost) { bestCost = cost; angle = a; }
    }
    return {
      pos: pos,
      star: { x: cx, y: cy },
      cx: cx, cy: cy, R: R, rings: rings, spacing: spacing, every: every, angle: angle,
      bounds: { x0: 2, y0: 2, x1: w - 2, y1: h - 2 },
      reserved: ringBoxes(angle)
    };
  }

  // One muted label per document near its own dots, clear of the dots, the star and each other.
  function placeLabels(m, L, widths) {
    var b = L.bounds;
    var midX = (b.x0 + b.x1) / 2, midY = (b.y0 + b.y1) / 2;
    var obstacles = L.reserved.slice();
    // Each dot with its rings, and its upper right, where its rank badge is tried first.
    L.pos.forEach(function (p) { obstacles.push({ x0: p.x - 11, x1: p.x + 19, y0: p.y - 18, y1: p.y + 11 }); });
    obstacles.push({ x0: L.star.x - 10, x1: L.star.x + 10, y0: L.star.y - 10, y1: L.star.y + 10 });
    var taken = [];
    return m.docs.map(function (doc, k) {
      var mine = L.pos.filter(function (p, i) { return m.points[i].doc === doc; });
      var cx = mine.reduce(function (s, p) { return s + p.x; }, 0) / mine.length;
      var cy = mine.reduce(function (s, p) { return s + p.y; }, 0) / mine.length;
      var away = Math.atan2(cy - midY, cx - midX);
      var half = widths[k] / 2 + 2;
      var best = null;
      for (var ring = 0; ring < 4; ring++) {
        for (var j = 0; j < 16; j++) {
          var ang = away + (j % 2 ? 1 : -1) * Math.ceil(j / 2) * Math.PI / 8;
          var x = cx + (16 + ring * 13) * Math.cos(ang), y = cy + (16 + ring * 13) * Math.sin(ang);
          var box = { x0: x - half, x1: x + half, y0: y - 7, y1: y + 7 };
          if (box.x0 < b.x0 || box.x1 > b.x1 || box.y0 < b.y0 || box.y1 > b.y1) continue;
          var cost = ring * 4 + Math.ceil(j / 2);
          obstacles.forEach(function (o) { if (overlaps(o, box)) cost += 30; });
          taken.forEach(function (o) { if (overlaps(o, box)) cost += 100; });
          if (!best || cost < best.cost) best = { cost: cost, x: x, y: y, box: box };
        }
      }
      if (!best) {
        var fx = clamp(cx, b.x0 + half, b.x1 - half), fy = clamp(cy, b.y0 + 7, b.y1 - 7);
        best = { x: fx, y: fy, box: { x0: fx - half, x1: fx + half, y0: fy - 7, y1: fy + 7 } };
      }
      taken.push(best.box);
      return { x: best.x, y: best.y, box: best.box };
    });
  }
  function textWidth(node, perChar) {
    var len = 0;
    try { len = node.getComputedTextLength(); } catch (e) { len = 0; }
    return len || node.textContent.length * perChar;
  }

  // A native radio group drawn as a segmented control.
  function radioGroup(name, label, options, current, onChange) {
    var group = el("div", "xp-segctl");
    group.setAttribute("role", "radiogroup");
    var caption = el("span", "xp-segctl__label", label);
    caption.id = name + "-label";
    group.setAttribute("aria-labelledby", caption.id);
    group.appendChild(caption);
    var opts = el("span", "xp-segctl__opts");
    options.forEach(function (o) {
      var wrap = el("label");
      var input = el("input");
      input.type = "radio";
      input.name = name;
      input.value = o.value;
      input.checked = o.value === current;
      input.addEventListener("change", function () { if (input.checked) onChange(o.value); });
      wrap.appendChild(input);
      wrap.appendChild(el("span", null, o.text));
      opts.appendChild(wrap);
    });
    group.appendChild(opts);
    return group;
  }

  // One debounced window listener redraws every map still on the page; maps whose section has
  // left the document (a new answer, another recorded run) are dropped, so nothing leaks.
  function watchResize(instance) {
    function live(item) { return document.documentElement.contains(item.node); }
    liveMaps = liveMaps.filter(live);
    liveMaps.push(instance);
    if (resizeBound) return;
    resizeBound = true;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        liveMaps = liveMaps.filter(live);
        liveMaps.forEach(function (item) { item.resize(); });
      }, 150);
    });
  }

  // A grid of coloured cells, one per number: blue positive, red negative, opacity by size.
  var gridCount = 0;
  function heatGrid(values, title, scaleWord, label) {
    gridCount += 1;
    var rows = Math.ceil(values.length / GRID_COLUMNS);
    var max = values.reduce(function (most, v) { return Math.max(most, Math.abs(v)); }, 0) || 1;
    var figure = el("figure", "xp-vm-grid");
    var caption = el("figcaption", null, title);
    caption.appendChild(el("span", "xp-vm-scale", "strongest " + scaleWord + " ±" + num(max, 4)));
    figure.appendChild(caption);
    var box = svg("svg", { viewBox: "0 0 " + GRID_COLUMNS + " " + rows, role: "img", "aria-label": label, preserveAspectRatio: "xMinYMin meet" }, "xp-vm-heat");
    var defs = svg("defs");
    var pattern = svg("pattern", { id: "xp-vm-cells-" + gridCount, width: 1, height: 1, patternUnits: "userSpaceOnUse" });
    pattern.appendChild(svg("rect", { x: 0.06, y: 0.06, width: 0.88, height: 0.88 }, "xp-vm-cellbg"));
    defs.appendChild(pattern);
    box.appendChild(defs);
    var full = Math.floor(values.length / GRID_COLUMNS);
    var fill = "url(#xp-vm-cells-" + gridCount + ")";
    if (full) box.appendChild(svg("rect", { x: 0, y: 0, width: GRID_COLUMNS, height: full, fill: fill }));
    if (values.length % GRID_COLUMNS) box.appendChild(svg("rect", { x: 0, y: full, width: values.length % GRID_COLUMNS, height: 1, fill: fill }));
    values.forEach(function (v, i) {
      if (!v) return;
      box.appendChild(svg("rect", {
        x: i % GRID_COLUMNS + 0.06, y: Math.floor(i / GRID_COLUMNS) + 0.06, width: 0.88, height: 0.88,
        "fill-opacity": (Math.abs(v) / max).toFixed(3)
      }, "xp-vm-cell " + (v > 0 ? "is-pos" : "is-neg")));
    });
    var hi = svg("rect", { x: 0, y: 0, width: 1, height: 1 }, "xp-vm-cellhi");
    box.appendChild(hi);
    figure.appendChild(box);
    return { figure: figure, svg: box, hi: hi, rows: rows };
  }

  function vectorMap(host, r, map) {
    var m = mapModel(r, map);
    var lenses = lensesFor(m);
    mapCount += 1;
    var uid = "xp-vm-" + mapCount;
    var start = m.quoted[0] || m.first("rerank") || m.first("dense") || m.bySim[0];
    var state = { view: "map", lens: "dense", pinned: start.id, hover: null };

    var section = el("section", "xp-vmap");
    section.setAttribute("aria-labelledby", uid + "-title");
    host.appendChild(section);
    var heading = el("h4", "xp-vm-title", "Where your question landed");
    heading.id = uid + "-title";
    section.appendChild(heading);
    section.appendChild(el("p", "xp-lede",
      "Every dot is one chunk of the corpus, placed by its embedding (" + m.dims + " numbers, flattened to two); the star is your question. " +
      "Colour and shape say which document a chunk comes from."));

    // ---- A. the map ----
    var controls = el("div", "xp-vm-controls");
    controls.appendChild(radioGroup(uid + "-view", "View", [
      { value: "map", text: "Map of the corpus" },
      { value: "dist", text: "Distance from the question" }
    ], state.view, function (value) { state.view = value; applyView(false); }));
    controls.appendChild(radioGroup(uid + "-lens", "Number the dots by", lenses.map(function (lens) {
      return { value: lens.prop, text: lens.label };
    }), state.lens, function (value) {
      state.lens = value;
      updateBadges();
      // Lines and leaders still wait for dots that are on their way.
      scheduleLinks(Math.max(0, movingUntil - Date.now()));
      updateCaption();
    }));
    section.appendChild(controls);

    // One column: the map, then the card (a tapped dot is read without scrolling), then the legend
    // and the caption. From about 860 px of width the card sits beside the map (explain.css).
    var stage = el("div", "xp-vm-stage");
    var figure = el("figure", "xp-vm-figure");
    var plot = svg("svg", {
      role: "application", tabindex: 0,
      "aria-label": "Map of " + plural(m.points.length, "chunk") + " from " + plural(m.docs.length, "document") +
        " and your question. The arrow keys step through the chunks from most to least similar to the question; Home and End jump to the ends.",
      // Focus reads the selected chunk, then the caption.
      "aria-describedby": uid + "-card " + uid + "-cap"
    }, "xp-vm-svg");
    figure.appendChild(plot);
    stage.appendChild(figure);
    var side = el("div", "xp-vm-side");
    // Not a live region: hover redraws it constantly. A deliberate choice is read out by "said".
    var card = el("div", "xp-vm-card");
    card.id = uid + "-card";
    side.appendChild(card);
    var said = el("p", "xp-sr");
    said.setAttribute("role", "status");
    side.appendChild(said);
    var hint = el("p", "xp-vm-hint", "Hover or tap a dot to read it; a click keeps it selected.");
    hint.appendChild(el("span", "xp-vm-hint__keys", " With the map focused, the arrow keys step through the chunks from most to least similar."));
    side.appendChild(hint);
    stage.appendChild(side);
    var keys = el("div", "xp-vm-keys");
    var legend = el("ul", "xp-vm-legend");
    keys.appendChild(legend);
    var caption = el("p", "xp-vm-mapcap");
    caption.id = uid + "-cap";
    keys.appendChild(caption);
    stage.appendChild(keys);
    section.appendChild(stage);
    var lesson = el("p", "xp-note xp-vm-lesson");
    section.appendChild(lesson);

    var how = el("details", "xp-vm-how");
    how.appendChild(el("summary", null, "How to read this map"));
    var howBody = el("div", "xp-vm-howbody");
    var neighbours = sameDocNeighbours(m);
    howBody.appendChild(el("p", null,
      "Your question and every chunk are lists of " + m.dims + " numbers: points in a space with " + m.dims + " directions, which no screen can show. " +
      "PCA (principal component analysis) finds the two directions in which the stored chunks differ most, and draws everything along those two. " +
      "Chunks of one document usually land together: they share subject and vocabulary, and each one was embedded with a short context that names its document. " +
      "A document that covers several topics can spread out, and documents on the same subject can overlap" +
      (!neighbours ? ". " : neighbours.same === neighbours.total
        ? "; on this map, every one of the " + neighbours.total + " chunks has a chunk of its own document as its nearest neighbour. "
        : "; on this map, " + neighbours.same + " of the " + neighbours.total + " chunks have a chunk of their own document as their nearest neighbour. ") +
      "The question is placed with the same two directions afterwards; it does not move them."));
    if (typeof m.variance[0] === "number" && typeof m.variance[1] === "number") {
      var v1 = Math.round(m.variance[0] * 1000) / 10, v2 = Math.round(m.variance[1] * 1000) / 10;
      var kept = Math.round((v1 + v2) * 10) / 10;
      howBody.appendChild(el("p", null,
        "On this map, direction 1 keeps " + v1.toFixed(1) + " % of the spread between the chunks and direction 2 keeps " + v2.toFixed(1) + " %: " +
        kept.toFixed(1) + " % in all. The other " + (100 - kept).toFixed(1) + " % is dropped."));
    }
    howBody.appendChild(el("p", null,
      "So two dots that sit close on the map can be far apart in the dimensions that were dropped. That is why the similarity numbers and the distance view exist: " +
      "every similarity, and every distance from the centre in the distance view, uses all " + m.dims + " numbers (the angles there still come from this map). " +
      "Search never uses this map; it compares the full vectors inside PostgreSQL."));
    howBody.appendChild(codeLinks(["app/retrieval/vector_map.py", "app/ingestion/contextualizer.py"]));
    how.appendChild(howBody);
    section.appendChild(how);

    // Legend: documents first (shape and colour), then only the markers this map uses.
    m.docs.filter(function (doc) { return doc.slot < 4; }).forEach(function (doc) {
      var li = el("li");
      li.appendChild(docKey(doc.slot));
      li.appendChild(document.createTextNode(docName(doc)));
      legend.appendChild(li);
    });
    function legendItem(icon, text) {
      var li = el("li");
      li.appendChild(icon);
      li.appendChild(document.createTextNode(text));
      legend.appendChild(li);
    }
    function dot() { return svg("path", { d: shapePath(0, 3.6) }, "xp-vm-shape xp-vm-shape--flat"); }
    if (m.docs.some(function (doc) { return doc.slot === 4; })) legendItem(docKey(4), "Other documents");
    legendItem(keyIcon(null, [svg("path", { d: starPath(7) }, "xp-vm-star")]), "your question");
    if (m.points.some(function (d) { return d.rerank && !d.cited; })) {
      legendItem(keyIcon("xp-vm-doc5", [svg("circle", { r: 7 }, "xp-vm-sent"), dot()]), "sent to the model");
    }
    if (m.points.some(function (d) { return d.cited; })) {
      legendItem(keyIcon("xp-vm-doc5", [svg("circle", { r: 7 }, "xp-vm-quoted"), dot()]), "quoted in the answer");
    }
    if (m.points.some(function (d) { return !d.fused && d.searched; })) {
      // A fused list cut at its limit may have dropped a chunk that one search arm did return.
      legendItem(keyIcon("xp-vm-doc5 is-hollow", [svg("path", { d: shapePath(0, 4.4) }, "xp-vm-shape")]),
        m.truncated ? "not among the fused candidates" : "not returned by either search");
    }
    if (m.points.some(function (d) { return !d.searched; })) {
      legendItem(keyIcon("xp-vm-doc5 is-hollow is-faint", [svg("path", { d: shapePath(0, 4.4) }, "xp-vm-shape")]), "filtered out (not searched)");
    }

    var lay = null, back = null, linkLayer = null, leaderLayer = null, starNode = null, linkTimer = null;
    // A change of view glides the dots for 0.6 s; until then nothing is re-ordered or drawn to them.
    var movingUntil = 0, settleTimer = null;
    var ptNodes = [], badgeNodes = [], labelNodes = [], leaders = [];
    var sizes = { map: 0, bump: 0 };

    function drawMap() {
      sizes.map = figure.clientWidth;
      var w = Math.max(260, Math.floor(sizes.map || 600));
      var h = mapHeight(m, w);
      clearTimeout(settleTimer);
      movingUntil = 0;
      // New dots are placed, not flown in from the corner: only a change of view animates.
      plot.classList.add("is-instant");
      plot.textContent = "";
      plot.setAttribute("viewBox", "0 0 " + w + " " + h);
      plot.setAttribute("width", w);
      plot.setAttribute("height", h);
      back = null;
      linkLayer = svg("g", null, "xp-vm-links");
      var labelLayer = svg("g", null, "xp-vm-labels");
      var pointLayer = svg("g", null, "xp-vm-points");
      leaderLayer = svg("g", null, "xp-vm-leaders");
      var badgeLayer = svg("g", null, "xp-vm-badges");
      [linkLayer, labelLayer, pointLayer, leaderLayer, badgeLayer].forEach(function (layer) { plot.appendChild(layer); });
      labelNodes = m.docs.map(function (doc) {
        var t = svgText(doc.code, { x: 0, y: 4, "text-anchor": "middle" }, "xp-vm-doclabel xp-vm-move");
        labelLayer.appendChild(t);
        return t;
      });
      ptNodes = m.points.map(function (d) {
        var g = svg("g", null, "xp-vm-pt xp-vm-move xp-vm-doc" + (d.doc.slot + 1) + (d.fused ? "" : " is-hollow") + (d.searched ? "" : " is-faint"));
        g.appendChild(svg("circle", { r: 13.5 }, "xp-vm-halo"));
        if (d.cited) g.appendChild(svg("circle", { r: 10 }, "xp-vm-quoted"));
        else if (d.rerank) g.appendChild(svg("circle", { r: 10 }, "xp-vm-sent"));
        g.appendChild(svg("path", { d: shapePath(d.doc.slot, 5) }, "xp-vm-shape"));
        pointLayer.appendChild(g);
        return g;
      });
      badgeNodes = m.points.map(function () {
        var t = svgText("", { x: 0, y: 0, "text-anchor": "middle" }, "xp-vm-badge xp-vm-move");
        badgeLayer.appendChild(t);
        return t;
      });
      starNode = svg("path", { d: starPath(8) }, "xp-vm-star xp-vm-move");
      plot.appendChild(starNode);
      var widths = labelNodes.map(function (t) { return textWidth(t, 6.6); });
      lay = { w: w, h: h, widths: widths, map: mapLayout(m, w, h), dist: null };
      lay.map.labels = placeLabels(m, lay.map, widths);
      applyView(true);
      markPoints();
      plot.getBoundingClientRect();
      plot.classList.remove("is-instant");
    }
    // The distance view is laid out the first time it is shown at this width: relaxing its dots is
    // the costly part, and a reader may never open it.
    function distLayout() {
      if (!lay.dist) {
        lay.dist = distanceLayout(m, lay.w, lay.h);
        lay.dist.labels = placeLabels(m, lay.dist, lay.widths);
      }
      return lay.dist;
    }

    function axisCaption(k) {
      var v = m.variance[k];
      if (typeof v !== "number") return "direction " + (k + 1);
      return "direction " + (k + 1) + " (" + Math.round(v * 100) + (k ? " %)" : " % of the spread)");
    }
    function drawBack() {
      if (back && back.parentNode) back.parentNode.removeChild(back);
      back = svg("g", null, "xp-vm-back");
      if (state.view === "map") {
        var M = lay.map;
        back.appendChild(svg("line", { x1: M.box.x0, y1: M.origin.y, x2: M.box.x1, y2: M.origin.y }, "xp-vm-axis"));
        back.appendChild(svg("line", { x1: M.origin.x, y1: M.box.y0, x2: M.origin.x, y2: M.box.y1 }, "xp-vm-axis"));
        back.appendChild(svgText(axisCaption(0), { x: (M.box.x0 + M.box.x1) / 2, y: lay.h - 8, "text-anchor": "middle" }, "xp-vm-cap"));
        back.appendChild(svgText(axisCaption(1), {
          transform: "translate(14 " + ((M.box.y0 + M.box.y1) / 2).toFixed(1) + ") rotate(-90)", "text-anchor": "middle"
        }, "xp-vm-cap"));
      } else {
        var D = distLayout();
        for (var t = 1; t <= D.rings; t++) {
          back.appendChild(svg("circle", { cx: D.cx, cy: D.cy, r: (D.spacing * t).toFixed(1) }, "xp-vm-ring"));
        }
        for (t = D.every; t <= D.rings; t += D.every) {
          back.appendChild(svgText(num(1 - t / 10, 1), {
            x: (D.cx + D.spacing * t * Math.cos(D.angle)).toFixed(1),
            y: (D.cy - D.spacing * t * Math.sin(D.angle) + 3.5).toFixed(1),
            "text-anchor": "middle"
          }, "xp-vm-ringlabel"));
        }
      }
      plot.insertBefore(back, plot.firstChild);
    }

    function current() { return state.view === "map" ? lay.map : distLayout(); }
    function lensOf(prop) { return lenses.filter(function (lens) { return lens.prop === prop; })[0]; }
    function lensTop(limit) {
      var key = state.lens;
      return m.points.filter(function (d) { return d[key]; })
        .sort(function (a, b) { return a[key] - b[key]; }).slice(0, limit);
    }
    // A rank badge sits next to its own dot: the first of eight directions, at two distances (just
    // outside the ring of a ringed dot), where the number is clearly nearer its own dot than any
    // other and covers no other dot, preferring places clear of labels, the star and earlier
    // badges. When no place qualifies, the number moves further out with a hairline to its dot.
    function updateBadges() {
      var L = current();
      var taken = L.reserved.concat(L.labels.map(function (label) { return label.box; }));
      taken.push({ x0: L.star.x - 10, x1: L.star.x + 10, y0: L.star.y - 10, y1: L.star.y + 10 });
      var dots = L.pos.map(function (p, i) {
        var r = ringed(m.points[i]) ? 11 : 7;
        return { x0: p.x - r, x1: p.x + r, y0: p.y - r, y1: p.y + r };
      });
      badgeNodes.forEach(function (node, i) {
        // Unused badges follow their dot, so one that appears later does not fly in from afar.
        node.textContent = "";
        moveTo(node, L.pos[i]);
      });
      leaders = [];
      lensTop(5).forEach(function (d) {
        var p = L.pos[d.i], text = String(d[state.lens]);
        var wide = text.length * 6.5 + 1;
        var near = ringed(d) ? [15, 19] : [11, 15];
        function spotAt(rad, a, pref) {
          var x = p.x + rad * Math.cos(a), y = p.y + rad * Math.sin(a);
          var box = { x0: x - wide / 2, x1: x + wide / 2, y0: y - 5.5, y1: y + 5.5 };
          if (box.x0 < 2 || box.x1 > lay.w - 2 || box.y0 < 2 || box.y1 > lay.h - 2) return null;
          var margin = Infinity, onDots = 0;
          L.pos.forEach(function (o, j) {
            if (j === d.i) return;
            margin = Math.min(margin, Math.sqrt(Math.pow(x - o.x, 2) + Math.pow(y - o.y, 2)) - rad);
            if (overlaps(dots[j], box)) onDots += 1;
          });
          var cost = pref + onDots * 5;
          taken.forEach(function (o) { if (overlaps(o, box)) cost += 10; });
          return { x: x, y: y, box: box, rad: rad, margin: margin, onDots: onDots, cost: cost };
        }
        function bestOf(radii, ok) {
          var best = null;
          radii.forEach(function (rad, ri) {
            BADGE_DIRS.forEach(function (a, k) {
              var s = spotAt(rad, a, ri * 2 + k * 0.1);
              if (s && ok(s) && (!best || s.cost < best.cost)) best = s;
            });
          });
          return best;
        }
        var spot = bestOf(near, function (s) { return s.margin >= 4 && !s.onDots; }) ||
          bestOf([24, 30], function (s) { return !s.onDots; }) ||
          bestOf(near.concat([24, 30]), function () { return true; }) ||
          { x: p.x, y: p.y - near[0], box: { x0: p.x - wide / 2, x1: p.x + wide / 2, y0: p.y - near[0] - 5.5, y1: p.y - near[0] + 5.5 }, rad: near[0] };
        taken.push(spot.box);
        var node = badgeNodes[d.i];
        node.textContent = text;
        moveTo(node, { x: spot.x, y: spot.y + 3.7 });
        if (spot.rad > near[1]) leaders.push({ p: p, box: spot.box, r: ringed(d) ? 11 : 7 });
      });
    }
    function drawLeaders() {
      leaderLayer.textContent = "";
      leaders.forEach(function (l) {
        // From the dot's edge to the nearest edge of its number.
        var bx = clamp(l.p.x, l.box.x0, l.box.x1), by = clamp(l.p.y, l.box.y0, l.box.y1);
        var len = Math.sqrt(Math.pow(bx - l.p.x, 2) + Math.pow(by - l.p.y, 2));
        if (len <= l.r + 2) return;
        leaderLayer.appendChild(svg("line", {
          x1: (l.p.x + (bx - l.p.x) * l.r / len).toFixed(1), y1: (l.p.y + (by - l.p.y) * l.r / len).toFixed(1),
          x2: bx.toFixed(1), y2: by.toFixed(1)
        }, "xp-vm-leader"));
      });
    }
    function scheduleLinks(delay) {
      clearTimeout(linkTimer);
      linkLayer.textContent = "";
      leaderLayer.textContent = "";
      // Lines appear once the dots have arrived, so they never point at a moving target.
      linkTimer = setTimeout(function () {
        drawLeaders();
        linkLayer.textContent = "";
        if (state.view !== "map") return;
        lensTop(3).forEach(function (d) {
          var p = lay.map.pos[d.i];
          linkLayer.appendChild(svg("line", {
            x1: lay.map.star.x.toFixed(1), y1: lay.map.star.y.toFixed(1), x2: p.x.toFixed(1), y2: p.y.toFixed(1)
          }, "xp-vm-link"));
        });
      }, delay);
    }
    function updateCaption() {
      var lens = lensOf(state.lens);
      var top = lensTop(5);
      var words = state.view === "map"
        ? "Same scale on both axes. Dots that sit together have similar embeddings, mostly: see “How to read this map”."
        : "Closer to the centre = more similar to the question: the distance from the centre is exact (1 − similarity, from all " + m.dims + " numbers). " +
          "The angle only places the dots: each chunk keeps the angle it has around the centre of the map view (where its axes cross), not around the question" +
          (m.points.length <= RELAX_MAX_DOTS ? ", nudged a little where dots would overlap" : "") +
          ". The gap between two dots does not show how alike they are.";
      words += top.length
        ? " Numbers: the top " + top.length + " by " + lens.arm + (state.view === "map" ? ", with lines from the question to the first " + Math.min(3, top.length) : "") + "."
        : " No chunk has a rank from " + lens.arm + ".";
      caption.textContent = words;
    }
    function applyView(initial) {
      var L = current();
      var wait = initial || reducedMotion() ? 0 : 650;
      movingUntil = Date.now() + wait;
      clearTimeout(settleTimer);
      // The point being read is raised above its neighbours once every dot has landed.
      if (wait) settleTimer = setTimeout(function () { movingUntil = 0; markPoints(); }, wait);
      ptNodes.forEach(function (node, i) { moveTo(node, L.pos[i]); });
      moveTo(starNode, L.star);
      labelNodes.forEach(function (node, k) { moveTo(node, L.labels[k]); });
      drawBack();
      updateBadges();
      scheduleLinks(wait);
      updateCaption();
      lesson.textContent = mapLesson(m, state.view);
      lesson.hidden = !lesson.textContent;
    }

    function nearestPoint(event) {
      var box = plot.getBoundingClientRect();
      if (!box.width || !lay) return null;
      var px = (event.clientX - box.left) * lay.w / box.width;
      var py = (event.clientY - box.top) * lay.h / box.height;
      var L = current();
      var best = null, bestDist = 22 * 22;
      m.points.forEach(function (d, i) {
        var dist = Math.pow(L.pos[i].x - px, 2) + Math.pow(L.pos[i].y - py, 2);
        if (dist < bestDist) { bestDist = dist; best = d; }
      });
      return best;
    }
    plot.addEventListener("pointermove", function (event) {
      var d = nearestPoint(event);
      plot.classList.toggle("is-pointing", !!d);
      preview(d ? d.id : null);
    });
    plot.addEventListener("pointerleave", function () { plot.classList.remove("is-pointing"); preview(null); });
    plot.addEventListener("click", function (event) { var d = nearestPoint(event); if (d) pick(d.id); });
    plot.addEventListener("keydown", function (event) {
      // Browser shortcuts (Alt+Left for back, Ctrl+Home, ...) pass through untouched.
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      var order = m.bySim;
      var at = order.indexOf(m.byId[state.pinned]);
      var next = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") next = Math.min(order.length - 1, at + 1);
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = Math.max(0, at - 1);
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = order.length - 1;
      if (next === null) return;
      event.preventDefault();
      pick(order[next].id);
    });

    function ranksText(d, sep) {
      return lenses.map(function (lens) { return lens.short + " " + rankText(d[lens.prop]); }).join(sep);
    }
    function renderCard(d) {
      card.textContent = "";
      var head = el("p", "xp-vm-card__doc");
      head.appendChild(docKey(d.doc.slot));
      head.appendChild(el("strong", null, d.doc.title));
      card.appendChild(head);
      card.appendChild(el("p", "xp-vm-card__where", "page " + d.page + " · chunk " + d.chunk + " of " + d.doc.chunks +
        (d.searched ? "" : " · filtered out, not searched")));
      card.appendChild(el("p", "xp-vm-card__sim", "similarity " + num(d.sim, 3)));
      card.appendChild(el("p", "xp-vm-card__ranks", ranksText(d, " · ")));
      if (d.score !== null) {
        card.appendChild(el("p", "xp-vm-card__score", m.noRerank
          ? "RRF score scaled to the top passage " + num(d.score, 2)
          : "cross-encoder score " + num(d.score, 2)));
      }
      if (d.cited || d.rerank) {
        var flag = el("p", "xp-vm-card__flag");
        flag.appendChild(keyIcon("xp-vm-doc5", [svg("circle", { r: 7 }, d.cited ? "xp-vm-quoted" : "xp-vm-sent"), dot()]));
        flag.appendChild(document.createTextNode(d.cited ? "quoted in the answer" : "sent to the model, not quoted"));
        card.appendChild(flag);
      }
      if (d.preview) card.appendChild(el("p", "preview", "“" + shorten(d.preview, 180) + "”"));
    }
    // What a screen reader hears after a deliberate choice: short, without the preview.
    function announce(d) {
      said.textContent = chunkName(d) + ", page " + d.page + (d.searched ? "" : ", filtered out") +
        ": similarity " + num(d.sim, 3) + "; " + (!d.fused ? "not a fused candidate" : lenses.map(function (lens) {
          return lens.short + " " + (d[lens.prop] ? "#" + d[lens.prop] : lens.prop === "rerank" ? "not sent" : "not returned");
        }).join(", ")) +
        (d.cited ? "; quoted in the answer." : d.rerank ? "; sent to the model." : ".");
    }

    // ---- selection, shared by the map, the bump chart, the chips and the table ----
    var bump = null, numbers = null, chunkTable = null;
    function markPoints() {
      var shownId = state.hover || state.pinned;
      var settled = Date.now() >= movingUntil;
      ptNodes.forEach(function (node, i) {
        var id = m.points[i].id;
        node.classList.toggle("is-selected", id === state.pinned);
        node.classList.toggle("is-hover", id === state.hover && id !== state.pinned);
        // The point being read is drawn last, above its neighbours.
        if (id === shownId && node.parentNode && node.parentNode.lastChild !== node) raise(node, settled);
      });
    }
    // Re-inserting a node cancels its glide, so during a change of view the dot is raised once it
    // has landed (the settle timer). moveBefore, where the browser has it, keeps the glide.
    function raise(node, settled) {
      var parent = node.parentNode;
      if (typeof parent.moveBefore === "function") {
        try { parent.moveBefore(node, null); return; } catch (e) { /* fall back below */ }
      }
      if (settled) parent.appendChild(node);
    }
    function refresh() {
      var d = m.byId[state.hover || state.pinned];
      markPoints();
      if (d) renderCard(d);
      if (bump) bump.focus(d);
      if (chunkTable) chunkTable.mark();
    }
    function preview(id) {
      if (id === state.hover) return;
      state.hover = id;
      refresh();
    }
    function pick(id) {
      state.pinned = id;
      state.hover = null;
      if (numbers) numbers.choose(id);
      refresh();
      if (m.byId[id]) announce(m.byId[id]);
    }

    // ---- B. the four rankings ----
    var partB = el("details", "xp-vm-part");
    partB.open = true;
    partB.appendChild(el("summary", null, "How the four rankings disagree"));
    var bumpBox = el("div", "xp-vm-bumpbox");
    partB.appendChild(bumpBox);
    var candidates = m.points.filter(function (d) { return d.fused; });
    var stopped = m.points.filter(function (d) { return d.searched && !d.fused; }).length;
    var filtered = m.points.filter(function (d) { return !d.searched; }).length;
    var maxRank = 1;
    candidates.forEach(function (d) { lenses.forEach(function (lens) { maxRank = Math.max(maxRank, d[lens.prop] || 0); }); });
    var rows = Math.min(10, maxRank);
    // What the bottom row holds, so the caption names only what is there.
    var lower = candidates.some(function (d) { return lenses.some(function (lens) { return d[lens.prop] > rows; }); });
    var unreturned = candidates.some(function (d) { return !d.dense || !d.sparse; });
    var notSent = candidates.some(function (d) { return !d.rerank; });
    var binned = lower || unreturned || notSent;
    if (candidates.length) {
      var keptCount = candidates.filter(function (d) { return d.rerank; }).length;
      var lastShort = lenses[3].short;
      var binParts = [];
      if (lower) binParts.push("ranks below #" + rows);
      if (unreturned) binParts.push("chunks a search arm did not return");
      partB.appendChild(el("p", "xp-note",
        "Each line is one chunk. Vector and Keyword are two separate searches run side by side, so a change between those two columns only means the two arms ranked the chunk differently. " +
        (m.noRerank
          ? "Fused merges both lists with RRF, and Kept is the fused order cut to the " + count(keptCount, "passage", "passages") + " sent to the model: no re-ranker ran. "
          : "Fused merges both lists with RRF, and Re-ranked is the cross-encoder's order of the " + count(keptCount, "passage", "passages") + " it sent to the model. " +
            "A line that climbs from Fused to Re-ranked was moved up by the cross-encoder. ") +
        (m.truncated
          ? "Only the " + plural(candidates.length, "fused candidate") + (candidates.length === 1 ? " is" : " are") + " drawn" +
            (stopped ? "; " + count(stopped, "other chunk", "other chunks") + " did not make the fused list, which keeps at most " + m.candidateLimit : "")
          : "Only the " + plural(candidates.length, "candidate") + " that a search arm returned " + (candidates.length === 1 ? "is" : "are") + " drawn" +
            (stopped ? "; " + count(stopped, "chunk was", "chunks were") + " returned by neither arm" : "")) +
        (filtered ? (stopped ? " and " : "; ") + count(filtered, "chunk was", "chunks were") + " filtered out before the search" : "") + "." +
        (binParts.length
          ? " The bottom row, “other”, holds " + andList(binParts) + (notSent ? "; in the " + lastShort + " column it holds the candidates not sent to the model" : "") + "."
          : notSent ? " In the " + lastShort + " column the bottom row, “other”, holds the candidates not sent to the model." : "")));
    }
    section.appendChild(partB);

    // Two-line column subtitles for narrow charts: "RRF," / "k = 60", "cross-" / "encoder".
    function splitSub(text) {
      var cut = text.indexOf(", ");
      if (cut > 0) return [text.slice(0, cut + 1), text.slice(cut + 2)];
      cut = text.indexOf(" ");
      if (cut > 0) return [text.slice(0, cut), text.slice(cut + 1)];
      cut = text.indexOf("-");
      if (cut > 0 && cut < text.length - 1) return [text.slice(0, cut + 1), text.slice(cut + 1)];
      return [text];
    }
    function drawBump() {
      sizes.bump = partB.clientWidth;
      bumpBox.textContent = "";
      if (!candidates.length) {
        bump = null;
        bumpBox.appendChild(el("p", "xp-note", "Neither search arm returned a chunk, so there are no rankings to compare."));
        return;
      }
      var w = Math.round(clamp(sizes.bump || 600, 260, 720));
      var rowH = 22, labelX = 30;
      var c0 = m.quoted[0];
      var chart = svg("svg", {
        role: "img",
        "aria-label": "Lines for the " + plural(candidates.length, "candidate chunk") + " through four rankings: " +
          andList(lenses.map(function (lens) { return lens.short.toLowerCase(); })) + "." +
          (c0 ? (m.quoted.length > 1 ? " The first quoted passage (" + chunkName(c0) + "): " : " The quoted passage (" + chunkName(c0) + "): ") +
            lenses.map(function (lens) { return lens.short.toLowerCase() + " " + rankText(c0[lens.prop]); }).join(", ") + "." : "")
      }, "xp-vm-bump");
      bumpBox.appendChild(chart);
      // The columns are evenly spaced (the chart is read by its slopes); when the headings do not
      // fit, their subtitles go on two lines. Headings are measured first, in the page.
      var heads = lenses.map(function (lens) {
        var head = svgText(lens.short, { y: 14, "text-anchor": "middle" }, "xp-vm-head");
        chart.appendChild(head);
        return { head: head, text: lens.prop === "fused" && m.rrfK ? "RRF, k = " + m.rrfK : lens.sub, subs: [], width: 0 };
      });
      function setSubs(lines) {
        heads.forEach(function (h) {
          h.subs.forEach(function (node) { chart.removeChild(node); });
          var parts = lines ? lines(h.text) : [h.text];
          h.subs = parts.map(function (text, k) {
            var node = svgText(text, { y: 28 + k * 12, "text-anchor": "middle" }, "xp-vm-sub");
            chart.appendChild(node);
            return node;
          });
          h.width = Math.max.apply(null, [textWidth(h.head, 6.6)].concat(h.subs.map(function (node) { return textWidth(node, 6); })));
        });
      }
      var colL, colR, even;
      function fits() {
        colL = Math.max(labelX + 14, heads[0].width / 2 + 2);
        colR = Math.min(w - 50, w - heads[3].width / 2 - 2);
        even = (colR - colL) / 3;
        return [1, 2, 3].every(function (c) { return (heads[c - 1].width + heads[c].width) / 2 + 8 <= even; });
      }
      setSubs(null);
      var top = 46;
      if (!fits()) { setSubs(splitSub); fits(); top = 58; }
      var binY = top + rows * rowH + 8;
      var h = binned ? binY + 30 : top + (rows - 1) * rowH + 18;
      chart.setAttribute("viewBox", "0 0 " + w + " " + h);
      chart.setAttribute("width", w);
      chart.setAttribute("height", h);
      function yOf(v) { return v && v <= rows ? top + (v - 1) * rowH : binY; }
      var xs = [0, 1, 2, 3].map(function (c) { return colL + c * even; });
      lenses.forEach(function (lens, c) {
        heads[c].head.setAttribute("x", xs[c].toFixed(1));
        heads[c].subs.forEach(function (node) { node.setAttribute("x", xs[c].toFixed(1)); });
        chart.appendChild(svg("line", { x1: xs[c].toFixed(1), y1: top - 10, x2: xs[c].toFixed(1), y2: (binned ? binY : yOf(rows)) + 10 }, "xp-vm-gridline"));
      });
      for (var k = 1; k <= rows; k++) chart.appendChild(svgText("#" + k, { x: labelX, y: yOf(k) + 4, "text-anchor": "end" }, "xp-vm-rank"));
      if (binned) {
        chart.appendChild(svg("line", { x1: 4, y1: binY - 15, x2: colR + 12, y2: binY - 15 }, "xp-vm-gridline"));
        chart.appendChild(svgText("other", { x: labelX, y: binY + 4, "text-anchor": "end" }, "xp-vm-rank"));
        if (notSent) {
          chart.appendChild(svgText("not sent", { x: xs[3], y: binY + 24, "text-anchor": "middle" }, "xp-vm-rank"));
        }
      }
      function linePoints(d) {
        return lenses.map(function (lens, c) { return xs[c].toFixed(1) + "," + yOf(d[lens.prop]).toFixed(1); }).join(" ");
      }
      function lineGroup(d, cls) {
        var g = svg("g", null, "xp-vm-bl xp-vm-doc" + (d.doc.slot + 1) + (cls ? " " + cls : ""));
        g.appendChild(svg("polyline", { points: linePoints(d) }, "xp-vm-bpath"));
        lenses.forEach(function (lens, c) {
          g.appendChild(svg("path", {
            d: shapePath(d.doc.slot, 4), transform: "translate(" + xs[c].toFixed(1) + " " + yOf(d[lens.prop]).toFixed(1) + ")"
          }, "xp-vm-shape"));
        });
        return g;
      }
      var thin = svg("g"), quoted = svg("g"), focus = svg("g", null, "xp-vm-bfocus"), labels = svg("g"), hits = svg("g");
      candidates.forEach(function (d) {
        (d.cited ? quoted : thin).appendChild(lineGroup(d, d.cited ? "is-quoted" : ""));
        if (d.cited) labels.appendChild(svgText("quoted", { x: xs[3] + 10, y: yOf(d.rerank) + 4 }, "xp-vm-endlabel"));
        var hit = svg("polyline", { points: linePoints(d) }, "xp-vm-bhit");
        hit.addEventListener("pointerenter", function () { preview(d.id); });
        hit.addEventListener("pointerleave", function () { preview(null); });
        hit.addEventListener("click", function () { pick(d.id); });
        hits.appendChild(hit);
      });
      [thin, quoted, focus, labels, hits].forEach(function (layer) { chart.appendChild(layer); });
      bump = {
        focus: function (d) {
          focus.textContent = "";
          if (d && d.fused) focus.appendChild(lineGroup(d, "is-focus"));
        }
      };
      bump.focus(m.byId[state.hover || state.pinned]);
    }

    // ---- C. the numbers behind one similarity ----
    function buildNumbers() {
      var q = m.query.vector || [];
      var spot = m.points.filter(function (d) { return q.length && d.vector && d.vector.length === q.length; });
      if (!spot.length) return null;
      spot.sort(function (a, b) {
        return (a.dense || Infinity) - (b.dense || Infinity) || (a.rerank || Infinity) - (b.rerank || Infinity) || a.i - b.i;
      });
      var part = el("details", "xp-vm-part");
      part.open = true;
      part.appendChild(el("summary", null, "The numbers behind the dots"));
      part.appendChild(el("p", null, "Your question became " + q.length + " numbers. The first six: " +
        q.slice(0, 6).map(function (v) { return num(v, 4); }).join(", ") + ", …"));
      var keyLine = el("p", "xp-vm-divkey");
      [["is-pos", "positive"], ["is-neg", "negative"]].forEach(function (pair) {
        var item = el("span");
        item.appendChild(el("span", "xp-vm-sw " + pair[0]));
        item.appendChild(document.createTextNode(pair[1]));
        keyLine.appendChild(item);
      });
      keyLine.appendChild(el("span", null, GRID_COLUMNS + " numbers to a row; the stronger the colour, the larger the number compared with the largest in its grid"));
      part.appendChild(keyLine);
      var chips = el("div", "xp-vm-chips");
      chips.setAttribute("role", "radiogroup");
      chips.setAttribute("aria-label", "Chunk to compare with the question");
      var inputs = {};
      spot.forEach(function (d) {
        var wrap = el("label", "xp-vm-chip");
        var input = el("input");
        input.type = "radio";
        input.name = uid + "-compare";
        input.value = d.id;
        input.addEventListener("change", function () { if (input.checked) pick(d.id); });
        var face = el("span");
        face.appendChild(docKey(d.doc.slot));
        var parts = [chunkName(d), d.dense ? "Vector #" + d.dense : "Vector –"];
        if (d.rerank) parts.push((m.noRerank ? "kept #" : "re-ranked #") + d.rerank);
        if (d.cited) parts.push("quoted");
        face.appendChild(document.createTextNode(parts.join(" · ")));
        wrap.appendChild(input);
        wrap.appendChild(face);
        chips.appendChild(wrap);
        inputs[d.id] = input;
      });
      part.appendChild(chips);
      var grids = el("div", "xp-vm-grids");
      part.appendChild(grids);
      var readout = el("p", "xp-vm-readout");
      part.appendChild(readout);
      var sums = el("div", "xp-vm-sums");
      part.appendChild(sums);
      var shown = null;
      // Dimensions on which the question and every chunk vector this answer carries have the same
      // sign and a size of at least 0.1: a direction these vectors share whatever their words say.
      var vecs = [q].concat(spot.map(function (d) { return d.vector; }));
      var shared = [];
      q.forEach(function (v, i) {
        if (vecs.every(function (vec) { return Math.abs(vec[i]) >= 0.1 && (vec[i] > 0) === (v > 0); })) shared.push(i);
      });
      function sharedPart(vec) { return shared.reduce(function (s, i) { return s + q[i] * vec[i]; }, 0); }
      var sharedParts = spot.map(function (d) { return sharedPart(d.vector); });
      var spotSums = spot.map(function (d) { return d.vector.reduce(function (s, v, i) { return s + v * q[i]; }, 0); });
      function spread(values) { return Math.max.apply(null, values) - Math.min.apply(null, values); }
      // "Steady": the shared part differs between these chunks by under a quarter of what their sums do.
      var steady = spot.length > 1 && spread(sharedParts) < 0.25 * spread(spotSums);

      function choose(id) {
        var d = m.byId[id];
        if (!d || !inputs[id]) return;
        inputs[id].checked = true;
        if (shown === id) return;
        shown = id;
        var c = d.vector;
        var products = q.map(function (v, i) { return v * c[i]; });
        var cells = [
          heatGrid(q, "Question", "number", "The question's " + q.length + " numbers"),
          heatGrid(c, "Chunk: " + chunkName(d), "number", "The " + c.length + " numbers of " + chunkName(d)),
          heatGrid(products, "Question × chunk", "product", "The " + q.length + " products of question and chunk, number by number")
        ];
        grids.textContent = "";
        readout.textContent = "Point at a cell to read its numbers.";
        function show(event) {
          var box = event.currentTarget.getBoundingClientRect();
          if (!box.width) return;
          var col = Math.floor((event.clientX - box.left) / box.width * GRID_COLUMNS);
          var row = Math.floor((event.clientY - box.top) / box.height * cells[0].rows);
          var i = row * GRID_COLUMNS + col;
          if (col < 0 || col >= GRID_COLUMNS || row < 0 || i >= q.length) { hide(); return; }
          readout.textContent = "dimension " + (i + 1) + ": question " + num(q[i], 4) + " × chunk " + num(c[i], 4) + " = " +
            num(products[i], Math.abs(products[i]) < 0.0001 ? 6 : 4);
          cells.forEach(function (g) {
            g.hi.setAttribute("x", col);
            g.hi.setAttribute("y", row);
            g.hi.classList.add("is-on");
          });
        }
        function hide() { cells.forEach(function (g) { g.hi.classList.remove("is-on"); }); }
        cells.forEach(function (g) {
          g.svg.addEventListener("pointermove", show);
          g.svg.addEventListener("pointerdown", show);
          g.svg.addEventListener("pointerleave", hide);
          grids.appendChild(g.figure);
        });

        var dotSum = 0, qq = 0, cc = 0;
        products.forEach(function (v, i) { dotSum += v; qq += q[i] * q[i]; cc += c[i] * c[i]; });
        var lq = Math.sqrt(qq), lc = Math.sqrt(cc);
        var cosine = dotSum / (lq * lc);
        var unit = Math.abs(lq - 1) <= 0.01 && Math.abs(lc - 1) <= 0.01;
        var shownSim = unit ? dotSum : cosine;
        var simText = num(shownSim, 3);
        // Read back the digits on screen (num writes a true minus sign), so "1 − s = d" adds up as shown.
        var simShown = Number(simText.replace("−", "-"));
        var distance = "pgvector's [[<=>]] returns 1 − " + (simShown < 0 ? "(" + simText + ")" : simText) + " = " + num(1 - simShown, 3) + ".";
        var norm = "a vector's length, or norm, is the square root of the sum of its squared numbers, not how many numbers it holds";
        var first;
        if (unit) {
          first = "Multiply the two lists number by number and add up the " + q.length + " products: " + num(dotSum, 3) + ". " +
            (num(lq, 3) === num(lc, 3) ? "Both vectors have length " + num(lq, 3) : "The two vectors have lengths " + num(lq, 3) + " and " + num(lc, 3) + ", 1 up to rounding") +
            " (" + norm + "). Cosine similarity is that sum divided by the two lengths, so here the sum itself is the cosine similarity the vector arm ranks by; " + distance;
        } else {
          first = "Multiply the two lists number by number and add up the " + q.length + " products: " + num(dotSum, 3) + ". " +
            "Divide by the two vectors' lengths, or norms (the square root of the sum of each one's squared numbers: " + num(lq, 3) + " and " + num(lc, 3) +
            ") and you get the cosine similarity the vector arm ranks by, " + simText + "; " + distance;
        }
        if (simText !== num(d.sim, 3)) {
          first += " The similarity on the map, " + num(d.sim, 3) + ", was computed before these numbers were rounded to four decimals.";
        }
        // How the sum is made up, computed: the share of the ten largest products, the largest one,
        // and how few dimensions make half of it.
        var order = products.map(function (v, i) { return i; }).sort(function (a, b) { return products[b] - products[a] || a - b; });
        var topSum = order.slice(0, 10).reduce(function (s, i) { return s + products[i]; }, 0);
        var against = products.filter(function (v) { return v < 0; }).length;
        var second;
        if (topSum > dotSum) {
          // The ten largest can exceed the whole sum when other dimensions disagree (negative products).
          second = "The ten largest products add up to " + num(topSum, 2) + ", more than the total of " + num(dotSum, 2) + ": " +
            count(against, "dimension disagrees and pulls", "dimensions disagree and pull") + " the sum back down with a negative product. " +
            "No single number means anything a person could name.";
        } else if (topSum < 0.25 * dotSum) {
          second = "The ten largest products add up to " + num(topSum, 2) + " of the " + num(dotSum, 2) +
            ": similarity is the sum of many small agreements, and no single number means anything a person could name.";
        } else {
          var half = 0, reach = 0;
          while (reach < order.length && half < dotSum / 2) { half += products[order[reach]]; reach += 1; }
          second = "The ten largest products add up to " + num(topSum, 2) + " of the " + num(dotSum, 2) + ", and dimension " + (order[0] + 1) +
            " alone gives " + num(products[order[0]], 2) + ": half of the sum comes from just " + reach + " of the " + q.length + " dimensions. " +
            "No single number means anything a person could name.";
        }
        sums.textContent = "";
        sums.appendChild(para(null, [first]));
        sums.appendChild(para(null, [second]));
        if (shared.length) {
          // Name the three that add most to this similarity, in dimension order (1-based, as the readout).
          var names = andList(shared.slice().sort(function (a, b) { return products[b] - products[a] || a - b; }).slice(0, 3)
            .sort(function (a, b) { return a - b; }).map(function (i) { return String(i + 1); }));
          var k = shared.length;
          var lo = Math.min.apply(null, sharedParts), hi = Math.max.apply(null, sharedParts);
          var sims = m.points.map(function (pt) { return pt.sim; });
          sums.appendChild(para(null, [
            "In the question and in " + (spot.length === 1 ? "the one chunk vector" : "all " + count(spot.length, "", "chunk vectors")) + " this answer carries, " +
            (k === 1 ? "dimension " + names + " has" : k <= 3 ? "dimensions " + names + " have" : count(k, "dimension", "dimensions") + " have") +
            " the same sign in every one of them and a size of at least 0.1" + (k > 3 ? ", among them " + names : "") + ". " +
            "Their products add " + num(sharedPart(c), 2) + " to this similarity of " + num(dotSum, 2) + ", whatever the words say: these vectors all lean the same way along " + (k === 1 ? "it" : "them") + ". " +
            (steady ? "They add " + (num(lo, 2) === num(hi, 2) ? "about " + num(lo, 2) : "between " + num(lo, 2) + " and " + num(hi, 2)) +
              " to the similarity of each of these chunks, so what tells the chunks apart is mostly the other " + (q.length - k) + " dimensions. " : "") +
            "A similarity is best read next to the others for the same question: on this map they run from " +
            num(Math.min.apply(null, sims), 2) + " to " + num(Math.max.apply(null, sims), 2) + ". " +
            "PCA subtracts the average chunk vector before it looks for directions (that is what centring is), so this shared lean does not move the dots on the map."
          ]));
        }
      }
      section.appendChild(part);
      var initial = (m.quoted[0] && inputs[m.quoted[0].id] && m.quoted[0]) || m.first("rerank") || m.first("dense");
      choose(initial && inputs[initial.id] ? initial.id : spot[0].id);
      return { choose: choose };
    }

    // ---- D. the table: the accessible view of both charts, and a sandbox ----
    function buildTable() {
      var part = el("details", "xp-vm-part");
      part.appendChild(el("summary", null, "All " + plural(m.points.length, "chunk") + " as a table"));
      part.appendChild(el("p", "xp-note", "Sort by each column to see how differently the stages rank the same chunks. A dash means that stage did not return the chunk" +
        (m.truncated ? "; the search ranks of a chunk that did not make the fused list of " + m.candidateLimit + " are not kept, so it shows dashes too." : ".")));
      var columns = [
        { key: "i", label: "Chunk", dir: 1 },
        { key: "sim", label: "Similarity", dir: -1 }
      ].concat(lenses.map(function (lens) { return { key: lens.prop, label: lens.short, dir: 1 }; }));
      var sort = { key: "sim", dir: -1 };
      var wrap = el("div", "tablewrap");
      var t = el("table", "xp-table xp-vm-table");
      var thead = el("thead");
      var headRow = el("tr");
      var ths = columns.map(function (col) {
        var th = el("th");
        th.scope = "col";
        var button = el("button", null, col.label);
        button.type = "button";
        button.addEventListener("click", function () {
          sort = { key: col.key, dir: sort.key === col.key ? -sort.dir : col.dir };
          render();
        });
        th.appendChild(button);
        headRow.appendChild(th);
        return th;
      });
      thead.appendChild(headRow);
      t.appendChild(thead);
      var tbody = el("tbody");
      t.appendChild(tbody);
      wrap.appendChild(t);
      part.appendChild(wrap);
      var rowsById = {};

      function compare(a, b) {
        var k = sort.key;
        if (k === "i" || k === "sim") return (a[k] - b[k]) * sort.dir || a.i - b.i;
        // A stage that did not return a chunk sorts it last, whichever way the column runs.
        if (!a[k] || !b[k]) return (a[k] ? -1 : 0) + (b[k] ? 1 : 0) || a.i - b.i;
        return (a[k] - b[k]) * sort.dir || a.i - b.i;
      }
      function render() {
        ths.forEach(function (th, c) {
          if (columns[c].key === sort.key) th.setAttribute("aria-sort", sort.dir > 0 ? "ascending" : "descending");
          else th.removeAttribute("aria-sort");
        });
        tbody.textContent = "";
        rowsById = {};
        m.points.slice().sort(compare).forEach(function (d) {
          var tr = el("tr");
          var cell = el("td");
          var button = el("button", "xp-vm-pick");
          button.type = "button";
          button.appendChild(docKey(d.doc.slot));
          button.appendChild(document.createTextNode(d.doc.code + " · p. " + d.page + " · chunk " + d.chunk));
          button.addEventListener("click", function () { pick(d.id); });
          cell.appendChild(button);
          var tags = [];
          if (d.cited) tags.push("quoted in the answer");
          else if (d.rerank) tags.push("sent to the model");
          if (!d.searched) tags.push("filtered out");
          if (tags.length) cell.appendChild(el("span", "xp-vm-tag", tags.join(" · ")));
          tr.appendChild(cell);
          tr.appendChild(el("td", "num", num(d.sim, 3)));
          lenses.forEach(function (lens) { tr.appendChild(el("td", "num", rankText(d[lens.prop]))); });
          tbody.appendChild(tr);
          rowsById[d.id] = { tr: tr, button: button };
        });
        mark();
      }
      function mark() {
        Object.keys(rowsById).forEach(function (id) {
          rowsById[id].tr.classList.toggle("is-selected", id === state.pinned);
          rowsById[id].button.setAttribute("aria-pressed", id === state.pinned ? "true" : "false");
        });
      }
      section.appendChild(part);
      render();
      return { mark: mark };
    }

    drawMap();
    drawBump();
    numbers = buildNumbers();
    chunkTable = buildTable();
    refresh();
    watchResize({
      node: section,
      resize: function () {
        if (Math.abs(figure.clientWidth - sizes.map) >= 2) drawMap();
        if (Math.abs(partB.clientWidth - sizes.bump) >= 2) drawBump();
      }
    });
    // What the steps below may point back to.
    return { numbers: !!numbers, rankings: candidates.length > 0 };
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
      term: "Vector map (PCA)",
      short: "A flat picture of the 384-dimensional space the search works in.",
      body: "No screen shows 384 directions. Principal component analysis (PCA) finds the two directions in which the stored chunks differ most and draws every chunk, and then the question, along those two. Chunks of one document usually land together, because they share subject and vocabulary; a document that covers several topics can spread out. Everything else is dropped, and the walkthrough says how much for each answer, so two dots that look close can be far apart in the dropped directions. That is why the walkthrough also has a distance view, which plots the exact cosine similarity of every chunk, and why search itself never uses the map.",
      code: ["app/retrieval/vector_map.py"]
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
      term: "Similarity is not relevance",
      short: "The nearest vector is not always the passage that answers.",
      body: "Vector search ranks chunks by how close their meaning is to the question, and it always returns its nearest chunks, even for a question the documents cannot answer. Close in meaning is not the same as answering: a chunk about the same subject can outrank the one clause that settles the question. Keyword search catches exact terms and identifiers that the embedding blurs, rank fusion rewards the chunks both arms agree on, and the cross-encoder reads question and passage together to judge whether the passage answers. Each corrects a different miss; the walkthrough's bump chart shows where they disagreed for one question.",
      code: ["app/retrieval/hybrid_search.py", "app/retrieval/reranker.py"]
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
