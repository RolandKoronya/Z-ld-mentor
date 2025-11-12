// lib/prompt.js
import fs from "fs";
import path from "path";

const PROMPT_PATH = path.join(process.cwd(), "prompts", "base.hu.md");

// Simple in-memory cache
let CACHE = { text: "", mtimeMs: 0 };

function readPromptFile() {
  try {
    const stat = fs.statSync(PROMPT_PATH);
    if (!CACHE.text || stat.mtimeMs !== CACHE.mtimeMs) {
      CACHE.text = fs.readFileSync(PROMPT_PATH, "utf8");
      CACHE.mtimeMs = stat.mtimeMs;
      console.log("[prompts] Loaded base.hu.md (" + Math.round(CACHE.text.length / 1024) + " KB)");
    }
    return CACHE.text;
  } catch (e) {
    console.warn("[prompts] Could not read prompts/base.hu.md, using fallback.");
    return "Te a Zöld Mentor vagy — barátságos, pontos, gyakorlatias. Magyarul válaszolsz.";
  }
}

export function buildSystemPrompt() {
  return readPromptFile();
}

export function invalidatePromptCache() {
  CACHE = { text: "", mtimeMs: 0 };
  console.log("[prompts] Cache invalidated.");
}
