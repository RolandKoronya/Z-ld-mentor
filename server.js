// server.js
// Zöld Mentor — secure chat backend with per-session memory + external prompts + KB (RAG)

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
import OpenAI from "openai";
import fs from "fs";
import path from "path";
import zlib from "zlib";

// ⤵️ New imports for the hybrid KB retriever
import { loadKB } from "./lib/kb_loader.js";
import { createRetriever } from "./lib/retriever.js";

// ─────────────────────────────────────────────────────────────────────────────
// 0) Boot
// ─────────────────────────────────────────────────────────────────────────────
dotenv.config();

const app = express();
app.set("trust proxy", 1);
app.use(express.json({ limit: "1mb" }));

// CORS: only allow your sites
const allowedOrigins = [
  "http://localhost:5173",
  "http://localhost:3000",
  "https://academiaeherba.hu",
  "https://www.academiaeherba.hu",
  "https://theherbalconservatory.eu",
  "https://www.theherbalconservatory.eu",
];
app.use(
  cors({
    origin(origin, callback) {
      if (!origin) return callback(null, true);
      if (allowedOrigins.includes(origin)) return callback(null, true);
      return callback(new Error("Not allowed by CORS"));
    },
  })
);

// Rate limit
const limiter = rateLimit({ windowMs: 60 * 1000, max: 120 });
app.use(limiter);

// Health check
app.get("/health", (_req, res) => res.json({ ok: true }));

// ─────────────────────────────────────────────────────────────────────────────
// 1) Auth
// ─────────────────────────────────────────────────────────────────────────────
const PUBLIC_API_TOKEN =
  process.env.PUBLIC_API_TOKEN || "zoldmentor-demo-1234567890";

function auth(req, res, next) {
  const authHeader = req.headers.authorization || "";
  const bearer = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : "";
  const alt = req.headers["x-client-token"] || "";
  const token = bearer || alt;
  const matches = token && token === PUBLIC_API_TOKEN;
  if (!matches) return res.status(401).json({ error: "Unauthorized" });
  return next();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2) OpenAI client
// ─────────────────────────────────────────────────────────────────────────────
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ─────────────────────────────────────────────────────────────────────────────
// 3) External prompt loader
// ─────────────────────────────────────────────────────────────────────────────
const PROMPT_PATH =
  process.env.PROMPT_PATH ||
  path.join(process.cwd(), "prompts", "base.hu.md");

let cachedSystemPrompt = null;
let cachedPromptMtime = 0;

function readFileIfExists(p) {
  try {
    return fs.readFileSync(p, "utf8");
  } catch {
    return "";
  }
}

function buildSystemPrompt() {
  try {
    const stat = fs.statSync(PROMPT_PATH);
    if (!cachedSystemPrompt || stat.mtimeMs !== cachedPromptMtime) {
      cachedSystemPrompt = readFileIfExists(PROMPT_PATH);
      cachedPromptMtime = stat.mtimeMs;
      console.log(
        `[PROMPT] Loaded base.hu.md (${PROMPT_PATH}, ${cachedSystemPrompt.length} chars)`
      );
    }
  } catch (e) {
    console.warn(`[PROMPT] Could not read ${PROMPT_PATH}: ${e.message}`);
    cachedSystemPrompt =
      cachedSystemPrompt ||
      "Te vagy a Zöld Mentor. Válaszolj magyarul, világosan.";
  }
  return cachedSystemPrompt;
}

app.post("/admin/reload-prompts", auth, (_req, res) => {
  cachedSystemPrompt = null;
  cachedPromptMtime = 0;
  const text = buildSystemPrompt();
  return res.json({ ok: true, length: text.length });
});

// ─────────────────────────────────────────────────────────────────────────────
// 4) Session memory
// ─────────────────────────────────────────────────────────────────────────────
const SESSIONS = new Map();
const MAX_HISTORY = 12;

function getSessionId(req) {
  return (req.headers["x-session-id"] || req.ip || "anon").toString();
}

function getHistory(sessionId) {
  if (!SESSIONS.has(sessionId)) SESSIONS.set(sessionId, []);
  return SESSIONS.get(sessionId);
}

function pushToHistory(sessionId, msg) {
  const arr = getHistory(sessionId);
  arr.push(msg);
  if (arr.length > MAX_HISTORY) arr.splice(0, arr.length - MAX_HISTORY);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5) NEW KB SYSTEM — hybrid retriever (replaces old searchKB)
// ─────────────────────────────────────────────────────────────────────────────
const kb = loadKB(path.join(process.cwd(), "kb"));
const retriever = createRetriever(kb, {
  openaiApiKey: process.env.OPENAI_API_KEY,
});

// Quick browser test: /search/debug?q=calendula
app.get("/search/debug", async (req, res) => {
  try {
    const q = req.query.q || "calendula";
    const hits = await retriever.search(q, { k: 6 });
    const shaped = hits.map((t) => ({
      source: t.source,
      score: Number(t.score.toFixed(4)),
      preview: t.text.length > 180 ? t.text.slice(0, 180) + "…" : t.text,
    }));
    res.json({ count: shaped.length, results: shaped });
  } catch (e) {
    console.error("❌ /search/debug error:", e.message);
    res.status(500).json({ error: "Search failed" });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 6) Helper to build system message from KB hits
// ─────────────────────────────────────────────────────────────────────────────
function buildKbSystemMessage(kbHits) {
  if (!kbHits || kbHits.length === 0) {
    return {
      role: "system",
      content:
        "NINCS ELÉRHETŐ KB-KONTEXTUS. Ha a kérdés speciális tudást igényel, mondd ki: 'nincs elég adat a tudástárban'.",
    };
  }
  const sourcesBlock = kbHits
    .map((h, i) => `#${i + 1} FORRÁS: ${h.source}\n${h.text}`)
    .join("\n\n---\n\n");

  return {
    role: "system",
    content: `KONTEKSTUS (KB-BÓL)\n${sourcesBlock}`,
  };
}

function buildKbScratchpad(kbHits) {
  if (!kbHits || kbHits.length === 0) return null;
  const lines = kbHits
    .map((h, i) => `#${i + 1} ${h.source} (score=${h.score.toFixed(3)})`)
    .join("\n");
  return {
    role: "assistant",
    content: `(SCRATCHPAD – ne idézd szó szerint)\nForrások:\n${lines}`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 7) Chat endpoint
// ─────────────────────────────────────────────────────────────────────────────
app.post("/chat", auth, async (req, res) => {
  try {
    const body = req.body || {};
    let incoming = Array.isArray(body.messages) ? body.messages : [];
    if (!incoming.length && body.message) {
      incoming = [{ role: "user", content: String(body.message) }];
    }
    if (!incoming.length)
      return res.status(400).json({ error: "Provide messages or message." });

    const lastUser = [...incoming].reverse().find((m) => m.role === "user");
    const userText = lastUser ? String(lastUser.content || "") : "";
    if (!userText)
      return res.status(400).json({ error: "Missing user message." });

    const sessionId = getSessionId(req);
    const history = getHistory(sessionId);

    // 🔍 Use the hybrid retriever instead of old searchKB
    const kbHits = await retriever.search(userText, { k: 6 });
    const kbSystem = buildKbSystemMessage(kbHits);
    const kbScratch = buildKbScratchpad(kbHits);

    const baseSystemPromptHu = buildSystemPrompt();

    const messages = [
      { role: "system", content: baseSystemPromptHu },
      kbSystem,
      ...(kbScratch ? [kbScratch] : []),
      ...history,
      ...incoming,
    ];

    const completion = await client.responses.create({
      model: "gpt-5",
      input: messages,
    });

    const reply =
      completion.output_text?.trim() ||
      completion.content?.trim() ||
      "nincs válasz";

    pushToHistory(sessionId, { role: "user", content: userText });
    pushToHistory(sessionId, { role: "assistant", content: reply });

    res.json({ ok: true, answer: reply });
  } catch (e) {
    console.error("❌ /chat error:", e);
    res.status(500).json({ error: "Error connecting to OpenAI" });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 8) Start server
// ─────────────────────────────────────────────────────────────────────────────
buildSystemPrompt();

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Zöld Mentor API listening on port ${PORT}`);
  console.log(`📂 KB loaded with ${kb.chunks.length} chunks`);
});
