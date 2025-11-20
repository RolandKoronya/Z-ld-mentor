// server.js
// Zöld Mentor — secure chat backend migrated to Google Gemini API

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
// ➡️ Import the GoogleGenAI client (Replaces OpenAI import)
import { GoogleGenAI } from "@google/genai";
import fs from "fs";
import path from "path";

// ⤵️ Imports for the hybrid KB retriever
import { loadKB } from "./lib/kb_loader.js";
import { createRetriever } from "./lib/retriever.js";

// ─────────────────────────────────────────────────────────────────────────────
// 0) Boot & Setup
// ─────────────────────────────────────────────────────────────────────────────
dotenv.config();

const app = express();
app.set("trust proxy", 1);
app.use(express.json({ limit: "1mb" }));

// CORS: only allow your sites (use the same list from your old server)
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
// 2) Gemini client and session management
// ─────────────────────────────────────────────────────────────────────────────
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.error("FATAL: GEMINI_API_KEY environment variable is not set.");
  process.exit(1);
}
const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

const SESSIONS = new Map();

function getSessionId(req) {
  // Use the same headers your frontend sends
  return (req.headers["x-session-id"] || req.ip || "anon").toString();
}

/**
 * Retrieves or creates a Gemini ChatSession for a given ID.
 * The session object internally manages history.
 * @param {string} sessionId
 * @param {string} systemInstructionText
 */
function getOrCreateChatSession(sessionId, systemInstructionText) {
  if (SESSIONS.has(sessionId)) {
    return SESSIONS.get(sessionId);
  }

  console.log(`Creating new Gemini ChatSession for ID: ${sessionId}`);

  // ➡️ Using gemini-2.5-pro for high-quality, complex reasoning tasks.
  const chat = ai.chats.create({
    model: "gemini-2.5-pro", 
    config: {
      systemInstruction: {
        parts: [{ text: systemInstructionText }],
      },
    }
  });

  SESSIONS.set(sessionId, chat);
  return chat;
}

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
// 4) KB SYSTEM & Helpers
// ─────────────────────────────────────────────────────────────────────────────
const kb = loadKB(path.join(process.cwd(), "kb"));
// ➡️ This uses the new Gemini-compatible retriever from lib/retriever.js
const retriever = createRetriever(kb, {
  // Pass the Gemini key, which the new retriever now expects as 'openaiApiKey' for compatibility
  openaiApiKey: process.env.GEMINI_API_KEY, 
});

function buildKbSystemMessage(kbHits) {
  if (!kbHits || kbHits.length === 0) {
    return ""; 
  }
  const sourcesBlock = kbHits
    .map((h, i) => `#${i + 1} FORRÁS: ${h.source}\n${h.text}`)
    .join("\n\n---\n\n");

  // This RAG context is inserted directly into the user's message
  return `KONTEKSTUS (KB-BÓL)\n${sourcesBlock}\n\n---\n\n`;
}

function buildKbScratchpad(kbHits) {
  if (!kbHits || kbHits.length === 0) return "";
  const lines = kbHits
    .map((h, i) => `#${i + 1} ${h.source} (score=${h.score.toFixed(3)})`)
    .join("\n");
  // This scratchpad is also inserted directly into the user's message
  return `(SCRATCHPAD – Források:\n${lines})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5) Chat endpoint
// ─────────────────────────────────────────────────────────────────────────────
app.post("/chat", auth, async (req, res) => {
  try {
    const body = req.body || {};
    let incoming = Array.isArray(body.messages) ? body.messages : [];
    
    // Fallback to reading body.message if messages array is empty
    if (!incoming.length && body.message) {
      incoming = [{ role: "user", content: String(body.message) }];
    }
    
    const lastUser = [...incoming].reverse().find((m) => m.role === "user");
    const userText = lastUser ? String(lastUser.content || "") : "";
    if (!userText)
      return res.status(400).json({ error: "Missing user message." });

    const sessionId = getSessionId(req);
    const baseSystemPromptHu = buildSystemPrompt();

    // 1. Get/Create the Gemini ChatSession with the system prompt
    const chat = getOrCreateChatSession(sessionId, baseSystemPromptHu);

    // 2. Perform RAG Search
    const kbHits = await retriever.search(userText, { k: 6 });
    const kbContext = buildKbSystemMessage(kbHits);
    const kbScratch = buildKbScratchpad(kbHits);

    // 3. Build the final prompt by prepending RAG context to the user's message
    const finalMessage = `${kbContext}${kbScratch}\n\nFelhasználó kérdése:\n${userText}`;

    // 4. Send the message to Gemini (history is managed internally by 'chat')
    const response = await chat.sendMessage({ message: finalMessage });

    const reply = response.text?.trim() || "nincs válasz";

    res.json({ ok: true, answer: reply });
  } catch (e) {
    console.error("❌ /chat error:", e);
    // ➡️ Update error message for Gemini
    res.status(500).json({ error: "Error connecting to Gemini API" });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 6) Start server
// ─────────────────────────────────────────────────────────────────────────────
buildSystemPrompt();

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Zöld Mentor API listening on port ${PORT}`);
  console.log(`📂 KB loaded with ${kb.chunks.length} chunks`);
});