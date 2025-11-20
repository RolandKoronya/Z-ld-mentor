// lib/retriever.js
// Migrated RAG system to use Gemini's Embedding Model (text-embedding-004)

import { GoogleGenAI } from "@google/genai";
import cosineSimilarity from "cosine-similarity";
// Assuming you are using a basic in-memory or file-based index structure
// where 'kb' is an object containing { chunks: [{ text: string, vector: number[] }] }

const EMBEDDING_MODEL = 'text-embedding-004'; // High-quality, multilingually capable model

/**
 * Creates a function to embed text using the Gemini API.
 * @param {object} aiClient - The initialized GoogleGenAI client instance.
 * @param {string} text - The text to embed.
 * @returns {Promise<number[]>} A vector array representing the text.
 */
async function embedText(aiClient, text) {
    // Implement simple exponential backoff for robustness
    for (let i = 0; i < 5; i++) {
        try {
            const response = await aiClient.models.embedContent({
                model: EMBEDDING_MODEL,
                content: { parts: [{ text: text }] },
            });
            // The API returns the embedding as a single vector array
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

/**
 * Re-indexes the Knowledge Base (KB) by generating new Gemini embeddings.
 * WARNING: This consumes API tokens! Only run when necessary (e.g., when updating KB files).
 * @param {object} kb - The knowledge base object from kb_loader.js.
 * @param {object} options - Options containing the API key.
 */
export async function reindexKB(kb, options) {
    // Note: The server.js passes the API key as 'openaiApiKey' for compatibility
    const apiKey = options.openaiApiKey; 

    if (!apiKey) {
        throw new Error("API Key missing for embedding generation.");
    }
    const ai = new GoogleGenAI({ apiKey });
    
    console.log(`[RAG] Starting re-indexing of ${kb.chunks.length} chunks using ${EMBEDDING_MODEL}...`);
    
    for (const chunk of kb.chunks) {
        try {
            const vector = await embedText(ai, chunk.text);
            chunk.vector = vector;
        } catch (e) {
            console.error(`[RAG] Failed to embed chunk: ${e.message}. Skipping.`);
            chunk.vector = null; 
        }
    }
    console.log("[RAG] Re-indexing complete.");
    // In a real app, you would save this new KB file with vectors to disk here.
}


/**
 * Creates the search retriever function.
 * @param {object} kb - The knowledge base object with chunks and vectors.
 * @param {object} options - Options containing the API key.
 * @returns {object} An object containing the search function.
 */
export function createRetriever(kb, options) {
    // Note: The server.js passes the API key as 'openaiApiKey' for compatibility
    const apiKey = options.openaiApiKey;

    if (!apiKey) {
        throw new Error("API Key missing for RAG initialization.");
    }
    const ai = new GoogleGenAI({ apiKey });

    return {
        /**
         * Searches the KB for chunks relevant to the query.
         * @param {string} query - The user's message text.
         * @param {object} params - Search parameters (e.g., k for top K results).
         * @returns {Promise<Array<object>>} The top K relevant chunks.
         */
        search: async (query, { k = 6 } = {}) => {
            if (!kb.chunks || kb.chunks.length === 0) return [];
            
            // 1. Embed the user's query
            const queryVector = await embedText(ai, query);

            // 2. Calculate similarity against all KB vectors
            const hits = kb.chunks
                .map(chunk => {
                    // Skip chunks that failed to embed during indexing
                    if (!chunk.vector) return null;
                    
                    // Use cosine similarity to find the best match
                    const score = cosineSimilarity(queryVector, chunk.vector);
                    
                    return {
                        ...chunk,
                        score: score
                    };
                })
                .filter(Boolean) // Remove null entries
                .sort((a, b) => b.score - a.score); // Sort by highest score first

            // 3. Return the top K results
            return hits.slice(0, k);
        }
    };
}