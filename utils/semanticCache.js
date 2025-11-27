const { pool } = require('./db');

const SIMILARITY_THRESHOLD = 0.92; // Độ giống 92% mới lấy

// Hàm 1: Tìm kiếm Cache bằng Vector
async function findInSemanticCache(queryVector) {
    try {
        // Format vector cho Postgres
        const vectorString = `[${queryVector.join(',')}]`;

        const query = `
            SELECT answer, 1 - (embedding <=> $1) as similarity
            FROM semantic_cache
            WHERE 1 - (embedding <=> $1) > $2
            ORDER BY similarity DESC
            LIMIT 1;
        `;
        const { rows } = await pool.query(query, [vectorString, SIMILARITY_THRESHOLD]);

        if (rows.length > 0) {
            console.log(`⚡ Hit Cache! Độ giống: ${(rows[0].similarity * 100).toFixed(1)}%`);
            return rows[0].answer;
        }
        return null;
    } catch (error) {
        console.error("⚠️ Lỗi Cache Check:", error.message);
        return null;
    }
}

// Hàm 2: Lưu Cache (Vector + Text)
async function saveToSemanticCache(question, answer, queryVector) {
    try {
        const vectorString = `[${queryVector.join(',')}]`;
        await pool.query(
            `INSERT INTO semantic_cache (question, answer, embedding) VALUES ($1, $2, $3)`,
            [question, answer, vectorString]
        );
    } catch (error) {
        console.error("⚠️ Lỗi Lưu Cache:", error.message);
    }
}

module.exports = { findInSemanticCache, saveToSemanticCache };