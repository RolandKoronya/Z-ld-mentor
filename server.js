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

// ─────────────────────────────────────────────────────────────────────────────
// 0) Boot
// ─────────────────────────────────────────────────────────────────────────────
dotenv.config();

const app = express();

// Running behind Render/NGINX → trust proxy so req.ip works
app.set("trust proxy", 1);

// Body parser
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
      if (!origin) return callback(null, true); // allow curl/Postman
      if (allowedOrigins.includes(origin)) return callback(null, true);
      return callback(new Error("Not allowed by CORS"));
    },
  })
);

// Basic rate limit (tune as needed)
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 120, // 120 req/min per IP
});
app.use(limiter);

// Health
app.get("/health", (_req, res) => res.json({ ok: true }));

// ─────────────────────────────────────────────────────────────────────────────
/** 1) Auth
 * Accepts:
 *  - Authorization: Bearer <PUBLIC_API_TOKEN>
 *  - OR X-Client-Token: <PUBLIC_API_TOKEN>
 */
const PUBLIC_API_TOKEN = process.env.PUBLIC_API_TOKEN || "zoldmentor-demo-1234567890";
function auth(req, res, next) {
  const authHeader = req.headers.authorization || "";
  const bearer = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : "";
  const alt = req.headers["x-client-token"] || "";
  const token = bearer || alt;

  const masked = token ? token.slice(0, 4) + "...(len=" + token.length + ")" : "none";
  const envSet = !!PUBLIC_API_TOKEN;
  const matches = token && token === PUBLIC_API_TOKEN;

  console.log(`Auth header received: ${masked}`);
  console.log(`Token from env (exists?): ${envSet}`);
  console.log(`Token matches stored: ${!!matches}`);

  if (!matches) return res.status(401).json({ error: "Unauthorized" });
  return next();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2) OpenAI client (Responses API will be used)
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ─────────────────────────────────────────────────────────────────────────────
/** 3) External prompt loader (prompts/base.hu.md)
 *  - buildSystemPrompt() returns the current text
 *  - /admin/reload-prompts to invalidate the cache
 */
const PROMPT_PATH = path.join(process.cwd(), "prompts", "base.hu.md");
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
      console.log(`[PROMPT] Loaded base.hu.md (${PROMPT_PATH}, ${cachedSystemPrompt.length} chars)`);
    }
  } catch (e) {
    console.warn(`[PROMPT] Could not read ${PROMPT_PATH}: ${e.message}`);
    cachedSystemPrompt = cachedSystemPrompt || "Te vagy a Zöld Mentor. Válaszolj magyarul, világosan.";
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
/** 4) Per-session memory (in-memory map)
 *  - Session is identified by 'X-Session-Id' (frontend should set it), else IP
 *  - Store last N messages to keep context light
 */
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
/** 5) KB / RAG: load shards, search, debug endpoint
 *  - Put your kb_store-000.json.gz .. 003 in repo root
 *  - Optionally set KB_DIR env to override dir
 */
const KB_DIR = process.env.KB_DIR || process.cwd();
const KB_GLOB_PREFIX = "kb_store-";
const KB_GLOB_SUFFIX = ".json.gz";

// Memory index: [{ id, text, source, embedding: number[] }]
let KB_INDEX = [];
let KB_SHARD_FILES = [];

function readGzipJsonArray(absPath) {
  const raw = fs.readFileSync(absPath);
  const buf = zlib.gunzipSync(raw);
  const arr = JSON.parse(buf.toString("utf-8"));
  if (!Array.isArray(arr)) throw new Error(`Shard is not a JSON array: ${absPath}`);
  return arr;
}

function loadKBShards() {
  try {
    const files = fs
      .readdirSync(KB_DIR)
      .filter((f) => f.startsWith(KB_GLOB_PREFIX) && f.endsWith(KB_GLOB_SUFFIX))
      .sort();

    KB_SHARD_FILES = files;

    if (files.length === 0) {
      console.warn(`⚠️ [KB] No ${KB_GLOB_PREFIX}*${KB_GLOB_SUFFIX} files found (dir=${KB_DIR})`);
      return;
    }

    const all = [];
    for (const f of files) {
      const abs = path.join(KB_DIR, f);
      try {
        const arr = readGzipJsonArray(abs);
        if (arr.length === 0) {
          console.warn(`⚠️ [KB] Shard has 0 chunks: ${f}`);
        }
        // Expect objects like { id, text, source, embedding: number[] }
        for (const item of arr) {
          if (item && item.text && Array.isArray(item.embedding)) {
            all.push(item);
          }
        }
      } catch (e) {
        console.error(`❌ [KB] Failed reading shard: ${f} — ${e.message}`);
      }
    }
    KB_INDEX = all;
    console.log(`[KB] Loaded ${KB_INDEX.length} chunks from ${files.length} shards (dir=${KB_DIR})`);
    console.log(`[KB] Shards: ${files.join(", ")}`);
  } catch (e) {
    console.error(`❌ [KB] load error: ${e.message}`);
  }
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) {
  return Math.sqrt(dot(a, a));
}
function cosine(a, b) {
  const na = norm(a),
    nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

async function searchKB(query, topK = 6) {
  if (!KB_INDEX.length) return [];
  const emb = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: query,
  });
  const qvec = emb.data[0].embedding;

  const scored = KB_INDEX.map((item) => ({
    item,
    score: cosine(qvec, item.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);

  return scored.slice(0, topK).map(({ item, score }) => ({
    source: item.source || "kb/unknown",
    score,
    text: item.text,
  }));
}

function buildKbSystemMessage(kbHits) {
  if (!kbHits || kbHits.length === 0) {
    return {
      role: "system",
      content:
        "NINCS ELÉRHETŐ KB-KONTEXTUS. Ha a kérdés speciális tudást igényel, mondd ki: 'nincs elég adat a tudástárban', és csak ezután fogalmazz meg óvatos, jelölt feltételezéseket.",
    };
  }
  const sourcesBlock = kbHits
    .map((h, i) => `#${i + 1} FORRÁS: ${h.source}\n${h.text}`)
    .join("\n\n---\n\n");

  return {
    role: "system",
    content: `KONTEKSTUS (KB-BÓL)
Az alábbi források **hiteles primer anyagok**. Szabályok:
- **Elsőbbség:** Először ezekből válaszolj. Ne találj ki új tényeket.
- **Ha nincs elég adat:** mondd ki expliciten: "nincs elég adat a tudástárban".
- **Spekuláció:** Csak jelölten („feltételezésem szerint…”) és minimálisan.
- **Terminológia:** Tartsd meg az eredeti magyar kifejezéseket és stílust.
- **Eltérés:** Ha a kérés túlnyúlik a forrásokon, jelezd világosan.

FORRÁSOK:
${sourcesBlock}`,
    };
}

function buildKbScratchpad(kbHits) {
  if (!kbHits || kbHits.length === 0) return null;
  const lines = kbHits
    .map((h, i) => `#${i + 1} ${h.source} (score=${h.score.toFixed(3)})`)
    .join("\n");
  return {
    role: "assistant",
    content: `(SCRATCHPAD – ne idézd szó szerint)
A válasz alapjául szolgáló források:
${lines}`,
  };
}

// Debug endpoint to peek at RAG results
app.post("/debug/rag", auth, async (req, res) => {
  try {
    const q = (req.body && req.body.q) ? String(req.body.q) : "";
    if (!q) return res.status(400).json({ error: "Missing q" });

    const top = await searchKB(q, 6);
    const shaped = top.map((t) => ({
      source: t.source,
      score: Number(t.score.toFixed(4)),
      preview: t.text.length > 200 ? t.text.slice(0, 200) + "…" : t.text,
    }));
    return res.json({ count: shaped.length, results: shaped });
  } catch (e) {
    console.error("❌ /debug/rag error:", e.message);
    return res.status(500).json({ error: "RAG debug failed" });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
/** 6) /chat — main endpoint
 *  - Builds messages: base system → KB system → (scratchpad) → history → user
 *  - Uses Responses API and lowers temperature when KB exists
 */
app.post("/chat", auth, async (req, res) => {
  try {
    const body = req.body || {};
    // Accept either { messages: [...] } or a simple { message: "..." }
    let incoming = Array.isArray(body.messages) ? body.messages : [];
    if (!incoming.length && body.message) {
      incoming = [{ role: "user", content: String(body.message) }];
    }
    if (!incoming.length) {
      return res.status(400).json({ error: "Provide messages or message." });
    }

    // Get the latest user message text
    const lastUser = [...incoming].reverse().find((m) => m.role === "user");
    const userText = lastUser ? String(lastUser.content || "") : "";
    if (!userText) {
      return res.status(400).json({ error: "Missing user message." });
    }

    // Session memory
    const sessionId = getSessionId(req);
    const history = getHistory(sessionId);

    // Retrieve KB hits and build strict KB messages
    const kbHits = await searchKB(userText, 6);
    const kbSystem = buildKbSystemMessage(kbHits);
    const kbScratch = buildKbScratchpad(kbHits);

    // Load base system prompt from prompts/base.hu.md
    const baseSystemPromptHu = buildSystemPrompt();

    // Construct the final messages in strict order
    const messages = [
      { role: "system", content: baseSystemPromptHu },
      kbSystem,
      ...(kbScratch ? [kbScratch] : []),
      ...history,
      ...incoming, // include any prior user/assistant turns sent by frontend (optional)
    ];

    // Ground harder if KB is present
    const temperature = kbHits.length ? 0.2 : 0.5;

    // Responses API call
    const completion = await client.responses.create({
      model: "gpt-4o",
      input: messages,
      temperature,
      max_output_tokens: 700,
    });

    // Extract text
    const reply = completion.output_text || "(nincs válasz)";

    // Push to memory (user + assistant turn)
    pushToHistory(sessionId, { role: "user", content: userText });
    pushToHistory(sessionId, { role: "assistant", content: reply });

    return res.json({ ok: true, answer: reply });
  } catch (e) {
    console.error("❌ /chat error:", e);
    return res.status(500).json({ error: "Error connecting to OpenAI" });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 7) Boot: load KB shards, preload prompt, start server
loadKBShards();
buildSystemPrompt();

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Zöld Mentor API listening on port ${PORT}`);
  console.log(`Working directory: ${process.cwd()}`);
});
