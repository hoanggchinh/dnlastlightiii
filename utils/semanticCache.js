const { pool } = require('./db');

const SIMILARITY_THRESHOLD = 0.95;
async function findInSemanticCache(userQuestion, queryVector) {
    try {

        const vectorString = `[${queryVector.join(',')}]`;
        const query = `
            SELECT answer, 1 - (embedding <=> $1) as similarity
            FROM semantic_cache
            WHERE 1 - (embedding <=> $1) > $2
            AND to_tsvector('simple', question) @@ plainto_tsquery('simple', $3)
            ORDER BY similarity DESC
            LIMIT 1;
        `;


        const { rows } = await pool.query(query, [vectorString, SIMILARITY_THRESHOLD, userQuestion]);

        if (rows.length > 0) {
            console.log(`Độ giống: ${(rows[0].similarity * 100).toFixed(1)}%`);
            return rows[0].answer;
        }

        console.log("Create new answer (Miss cache or keyword mismatch)");
        return null;
    } catch (error) {
        console.error("Lỗi Cache Check:", error.message);
        return null;
    }
}

async function saveToSemanticCache(question, answer, queryVector) {
    try {
        const vectorString = `[${queryVector.join(',')}]`;
        await pool.query(
            `INSERT INTO semantic_cache (question, answer, embedding) VALUES ($1, $2, $3)`,
            [question, answer, vectorString]
        );
        console.log("Đã lưu vào Cache.");
    } catch (error) {
        console.error("Lỗi Lưu Cache:", error.message);
    }
}

module.exports = { findInSemanticCache, saveToSemanticCache };