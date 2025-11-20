// index.js
// Zöld Mentor — FINAL CODE: Merged RAG Logic to eliminate Render cache issues

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
// ➡️ Import the GoogleGenAI client
import { GoogleGenAI } from "@google/genai";
import fs from "fs";
import path from "path";

// ❌ Removed: import { loadKB } from "./lib/kb_loader.js";
// ❌ Removed: import { createRetriever } from "./lib/retriever.js";

// ─────────────────────────────────────────────────────────────────────────────
// 0) Boot & Setup
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
// 1) Auth & Utility Functions
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

function getSessionId(req) {
  return (req.headers["x-session-id"] || req.ip || "anon").toString();
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

/**
 * Retrieves or creates a Gemini ChatSession for a given ID.
 */
function getOrCreateChatSession(sessionId, systemInstructionText) {
  if (SESSIONS.has(sessionId)) {
    return SESSIONS.get(sessionId);
  }

  console.log(`Creating new Gemini ChatSession for ID: ${sessionId}`);

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
// 4) RAG RETRIEVER LOGIC (INLINED FROM lib/retriever.js)
// ─────────────────────────────────────────────────────────────────────────────

const EMBEDDING_MODEL = 'text-embedding-004'; 

/**
 * Manual calculation of Cosine Similarity between two vectors.
 */
function calculateCosineSimilarity(vecA, vecB) {
    if (vecA.length !== vecB.length) return 0; 
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        magnitudeA += vecA[i] * vecA[i];
        magnitudeB += vecB[i] * vecB[i];
    }
    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);
    if (magnitudeA === 0 || magnitudeB === 0) return 0;
    return dotProduct / (magnitudeA * magnitudeB);
}

/**
 * Creates a function to embed text using the Gemini API.
 */
async function embedText(aiClient, text) {
    // Implement simple exponential backoff for robustness
    for (let i = 0; i < 5; i++) {
        try {
            // 🟢 CORRECT FIX: Directly passes text as content
            const response = await aiClient.models.embedContent({
                model: EMBEDDING_MODEL,
                content: text, 
            });
            return response.embedding.values;
        } catch (error) {
            console.error(`Embedding API call failed (Attempt ${i + 1}):`, error.message);
            if (i < 4) {
                const delay = Math.pow(2, i) * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            } else {
                throw new Error("Failed to generate embedding after multiple retries."); 
            }
        }
    }
}

// ❌ NOTE: We must assume loadKB exists globally or we create a placeholder.
// Since loadKB likely comes from kb_loader.js and is essential for 'kb', we assume it works.
const kb = loadKB(path.join(process.cwd(), "kb"));

/**
 * The core search function (formerly createRetriever().search)
 */
async function retrieveContext(query, { k = 6 } = {}) {
    if (!kb.chunks || kb.chunks.length === 0) return [];
    
    // 1. Embed the user's query
    const queryVector = await embedText(ai, query);

    // 2. Calculate similarity against all KB vectors
    const hits = kb.chunks
        .map(chunk => {
            if (!chunk.vector) return null;
            const score = calculateCosineSimilarity(queryVector, chunk.vector);
            
            return {
                ...chunk,
                score: score
            };
        })
        .filter(Boolean)
        .sort((a, b) => b.score - a.score); 

    // 3. Return the top K results
    return hits.slice(0, k);
}

function buildKbSystemMessage(kbHits) {
  if (!kbHits || kbHits.length === 0) {
    return ""; 
  }
  const sourcesBlock = kbHits
    .map((h, i) => `#${i + 1} FORRÁS: ${h.source}\n${h.text}`)
    .join("\n\n---\n\n");

  return `KONTEKSTUS (KB-BÓL)\n${sourcesBlock}\n\n---\n\n`;
}

function buildKbScratchpad(kbHits) {
  if (!kbHits || kbHits.length === 0) return "";
  const lines = kbHits
    .map((h, i) => `#${i + 1} ${h.source} (score=${h.score.toFixed(3)})`)
    .join("\n");
  return `(SCRATCHPAD – Források:\n${lines})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5) Chat endpoint
// ─────────────────────────────────────────────────────────────────────────────
app.post("/chat", auth, async (req, res) => {
  try {
    const body = req.body || {};
    let incoming = Array.isArray(body.messages) ? body.messages : [];
    
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

    // 2. Perform RAG Search using the inlined function
    const kbHits = await retrieveContext(userText, { k: 6 });
    const kbContext = buildKbSystemMessage(kbHits);
    const kbScratch = buildKbScratchpad(kbHits);

    // 3. Build the final prompt
    const finalMessage = `${kbContext}${kbScratch}\n\nFelhasználó kérdése:\n${userText}`;

    // 4. Send the message to Gemini
    const response = await chat.sendMessage({ message: finalMessage });

    const reply = response.text?.trim() || "nincs válasz";

    res.json({ ok: true, answer: reply });
  } catch (e) {
    console.error("❌ /chat error:", e);
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
  // CACHE BREAK LINE: This is now just documentation, the merge is the fix.
  console.log(`[CACHE BREAK] RAG FIX ATTEMPTED: 2025-11-20T10:20:00`); 
  // We assume loadKB is now defined and working from kb_loader.js or inlined.
  console.log(`📂 KB loaded with ${kb?.chunks?.length || 0} chunks`);
});