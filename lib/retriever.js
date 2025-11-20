// lib/retriever.js
// Migrated RAG system to use Gemini's Embedding Model (text-embedding-004)
// FIX: Corrected the input structure for aiClient.models.embedContent

import { GoogleGenAI } from "@google/genai";
// Removed external cosineSimilarity dependency

const EMBEDDING_MODEL = 'text-embedding-004'; // High-quality, multilingually capable model

/**
 * Manual calculation of Cosine Similarity between two vectors.
 * @param {number[]} vecA - Query vector.
 * @param {number[]} vecB - Chunk vector.
 * @returns {number} The similarity score (0 to 1).
 */
function calculateCosineSimilarity(vecA, vecB) {
    if (vecA.length !== vecB.length) {
        // This should not happen if all embeddings are generated correctly
        return 0; 
    }

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

    if (magnitudeA === 0 || magnitudeB === 0) {
        return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
}

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
            // 🟢 FIX: Correctly structure the content for the embedding API
            const response = await aiClient.models.embedContent({
                model: EMBEDDING_MODEL,
                content: text, // Pass the text directly as the content
            });
            
            // The API returns the embedding as a single vector array
            return response.embedding.values;
        } catch (error) {
            console.error(`Embedding API call failed (Attempt ${i + 1}):`, error.message);
            if (i < 4) {
                const delay = Math.pow(2, i) * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            } else {
                // The fatal error from the logs
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
}


/**
 * Creates the search retriever function.
 * @param {object} kb - The knowledge base object with chunks and vectors.
 * @param {object} options - Options containing the API key.
 * @returns {object} An object containing the search function.
 */
export function createRetriever(kb, options) {
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
                    
                    // Use the manual function instead of the imported package
                    const score = calculateCosineSimilarity(queryVector, chunk.vector);
                    
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