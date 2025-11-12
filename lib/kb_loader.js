// lib/kb_loader.js
import fs from "fs";
import path from "path";
import zlib from "zlib";

const stripAccents = (s) =>
  s.normalize("NFD").replace(/[\u0300-\u036f]/g, "");

export function normalize(s) {
  if (!s) return "";
  return stripAccents(String(s).toLowerCase());
}

/** Load all *.json.gz from /kb and return {chunks, inverted, avgdl} */
export function loadKB(kbDir = path.join(process.cwd(), "kb")) {
  const files = fs
    .readdirSync(kbDir)
    .filter((f) => f.endsWith(".json.gz"))
    .map((f) => path.join(kbDir, f));

  const chunks = [];
  for (const file of files) {
    const raw = fs.readFileSync(file);
    const jsonStr = zlib.gunzipSync(raw).toString("utf8");
    const arr = JSON.parse(jsonStr);
    for (const e of arr) {
      if (!e?.text || !e?.embedding) continue;
      const text = e.text;
      chunks.push({
        id: e.id ?? null,
        source: e.source ?? null,
        text,
        normText: normalize(text),
        embedding: e.embedding, // float[]
      });
    }
  }

  // Build tiny inverted index for keyword scoring
  const inverted = new Map();
  const tokenRe = /[a-z0-9]+/g;
  const docTokenCounts = new Array(chunks.length).fill(0);

  chunks.forEach((c, i) => {
    const tokens = (c.normText.match(tokenRe) || []).slice(0, 500);
    docTokenCounts[i] = tokens.length;
    const seen = new Map();
    for (const t of tokens) seen.set(t, (seen.get(t) || 0) + 1);
    for (const [t, tf] of seen) {
      if (!inverted.has(t)) inverted.set(t, []);
      inverted.get(t).push([i, tf]);
    }
  });

  const avgdl =
    docTokenCounts.reduce((a, b) => a + b, 0) / Math.max(1, chunks.length);

  return { chunks, inverted, avgdl };
}
