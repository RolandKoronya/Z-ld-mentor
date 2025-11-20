// tools/ingest.js (FAST SHARDING)
import fs from "fs";
import path from "path";
import zlib from "zlib";
import OpenAI from "openai";

const SRC_DIR = path.join(process.cwd(), "kb");
const OUT_PREFIX = "kb_store-"; // kb_store-000.json.gz, 001, ...

// Tweakables
const EMB_MODEL = "text-embedding-3-small"; // 1536-dim
const CHUNK_SIZE = 900;
const CHUNK_OVERLAP = 150;
const DECIMALS = 4;           // round embeddings for smaller files
const SHARD_COUNT_TARGET = 2500; // ~2500 chunks per shard (fast + well under 100MB gz)

// ——————————————————————————————————————

if (!process.env.OPENAI_API_KEY) {
  console.error("Missing OPENAI_API_KEY env var.");
  process.exit(1);
}

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function chunk(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  for (let i = 0; i < text.length; i += (size - overlap)) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

function loadTxtFiles() {
  if (!fs.existsSync(SRC_DIR)) fs.mkdirSync(SRC_DIR, { recursive: true });
  const files = fs.readdirSync(SRC_DIR).filter(f => f.endsWith(".txt"));
  const docs = [];
  for (const file of files) {
    const full = fs.readFileSync(path.join(SRC_DIR, file), "utf8");
    const parts = chunk(full);
    parts.forEach((c, i) => docs.push({ id: `${file}#${i}`, source: file, text: c }));
  }
  return docs;
}

function roundEmbedding(arr, decimals = DECIMALS) {
  const f = Math.pow(10, decimals);
  return arr.map(v => Math.round(v * f) / f);
}

function gzipJson(obj) {
  const json = JSON.stringify(obj);
  return zlib.gzipSync(Buffer.from(json));
}

function pad(n, width=3) {
  return String(n).padStart(width, "0");
}

async function main() {
  const docs = loadTxtFiles();
  if (docs.length === 0) {
    console.log("No .txt files found in /kb. Add some first.");
    return;
  }
  console.log(`Embedding ${docs.length} chunks with ${EMB_MODEL}...`);

  const out = [];
  const BATCH = 64;
  for (let i = 0; i < docs.length; i += BATCH) {
    const batch = docs.slice(i, i + BATCH);
    const resp = await openai.embeddings.create({
      model: EMB_MODEL,
      input: batch.map(d => d.text)
    });
    resp.data.forEach((row, j) => {
      out.push({
        id: batch[j].id,
        source: batch[j].source,
        text: batch[j].text,
        embedding: roundEmbedding(row.embedding)
      });
    });
    console.log(`  → ${Math.min(i + BATCH, docs.length)} / ${docs.length}`);
  }

  // FAST SHARDING: split by count, gzip once per shard
  let shardIdx = 0;
  for (let start = 0; start < out.length; start += SHARD_COUNT_TARGET) {
    const end = Math.min(start + SHARD_COUNT_TARGET, out.length);
    const shard = out.slice(start, end);
    const gz = gzipJson(shard);
    const fname = `${OUT_PREFIX}${pad(shardIdx)}.json.gz`;
    fs.writeFileSync(path.join(process.cwd(), fname), gz);
    console.log(`Saved ${fname} (${(gz.length/1024/1024).toFixed(1)} MB, ${shard.length} chunks)`);
    shardIdx += 1;
  }

  console.log("Done. Sharded and compressed KB ready.");
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});

