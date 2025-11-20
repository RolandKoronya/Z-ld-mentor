// lib/retriever.js
import OpenAI from "openai";
import { normalize } from "./kb_loader.js";

// Add/extend as needed
const SYNONYMS = {
  calendula: ["körömvirág", "marigold"],
  marigold: ["calendula", "körömvirág"],
  "körömvirág": ["calendula", "marigold"],
  yarrow: ["cickafark", "achillea"],
  "cickafark": ["yarrow", "achillea"],
  plantain: ["útifű", "plantago"],
  "útifű": ["plantain", "plantago"],
  ginger: ["gyömbér", "zingiber"],
  "gyömbér": ["ginger", "zingiber"],
};

function expandQuery(q) {
  const nq = normalize(q);
  const parts = nq.split(/\s+/).filter(Boolean);
  const bag = new Set(parts);
  for (const p of parts) {
    if (SYNONYMS[p]) for (const s of SYNONYMS[p]) bag.add(normalize(s));
  }
  return Array.from(bag);
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const L = Math.min(a.length, b.length);
  for (let i = 0; i < L; i++) {
    dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
  }
  if (!na || !nb) return 0;
  return dot / Math.sqrt(na * nb);
}

function bm25Lite({ tokens, inverted, avgdl, docLens, k1 = 1.2, b = 0.75 }) {
  const N = docLens.length;
  const df = new Map();
  const postings = new Map();
  for (const t of tokens) {
    const post = inverted.get(t);
    if (!post) continue;
    df.set(t, post.length);
    postings.set(t, post);
  }

  const scores = new Map();
  for (const t of tokens) {
    const post = postings.get(t);
    if (!post) continue;
    const dft = df.get(t) || 0;
    const idf = Math.log(1 + (N - dft + 0.5) / (dft + 0.5));
    for (const [docId, tf] of post) {
      const dl = docLens[docId] || 1;
      const denom = tf + k1 * (1 - b + (b * dl) / (avgdl || 1));
      const s = idf * ((tf * (k1 + 1)) / (denom || 1));
      scores.set(docId, (scores.get(docId) || 0) + s);
    }
  }
  return scores;
}

export function createRetriever(kb, { openaiApiKey, embedModel = "text-embedding-3-large" } = {}) {
  const client = new OpenAI({ apiKey: openaiApiKey });
  const docLens = kb.chunks.map((c) => (c.normText.match(/[a-z0-9]+/g) || []).length || 1);

  async function embedQuery(q) {
    const res = await client.embeddings.create({ model: embedModel, input: q });
    return res.data[0].embedding;
  }

  /** Hybrid search with robust fallbacks */
  async function search(query, { k = 12, k_kw = 80, alpha = 0.55 } = {}) {
    const expanded = expandQuery(query);
    const bm = bm25Lite({
      tokens: expanded,
      inverted: kb.inverted,
      avgdl: kb.avgdl,
      docLens,
    });

    // Preselect by keyword first
    const pre = [...bm.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, k_kw)
      .map(([docId]) => docId);

    // If no keyword hits, scan a subset of all docs
    let candidateIds = pre.length
      ? pre
      : [...Array(kb.chunks.length).keys()].slice(0, Math.min(2000, kb.chunks.length));

    // Query embedding
    let qEmb = null;
    try { qEmb = await embedQuery(query); } catch { qEmb = null; }

    const results = [];
    for (const id of candidateIds) {
      const ch = kb.chunks[id];
      const kwScore = bm.get(id) || 0;
      const semScore = qEmb ? cosine(qEmb, ch.embedding) : 0;
      const hybrid = alpha * semScore + (1 - alpha) * Math.tanh(kwScore);
      results.push({ id, hybrid, kwScore, semScore, chunk: ch });
    }

    results.sort((a, b) => b.hybrid - a.hybrid);
    let top = results.slice(0, k).map((r) => ({
      id: r.chunk.id,
      source: r.chunk.source,
      score: r.hybrid,
      kw: r.kwScore,
      sem: r.semScore,
      text: r.chunk.text,
    }));

    // Last-ditch fallback: naive substring search
    if (top.length === 0) {
      const nq = normalize(query);
      top = kb.chunks
        .map((c, i) => ({ i, hit: c.normText.includes(nq) }))
        .filter((h) => h.hit)
        .slice(0, k)
        .map(({ i }) => ({
          id: kb.chunks[i].id,
          source: kb.chunks[i].source,
          score: 0.01, kw: 0.01, sem: 0,
          text: kb.chunks[i].text,
        }));
    }

    return top;
  }

  return { search };
}
