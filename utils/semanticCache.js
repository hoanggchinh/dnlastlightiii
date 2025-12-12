const { pool } = require('./db');
const { normalizeForMatching, extractKeywords } = require('./sanitize');

const SIMILARITY_THRESHOLD = 0.92;
const KEYWORD_MATCH_THRESHOLD = 0.6;

async function findInSemanticCache(userQuestion, queryVector) {
    try {
        if (!queryVector || !Array.isArray(queryVector) || queryVector.length === 0) {
            return null;
        }

        if (!userQuestion || typeof userQuestion !== 'string') {
            return null;
        }

        const vectorString = `[${queryVector.join(',')}]`;
        const normalizedQuestion = normalizeForMatching(userQuestion);
        const userKeywords = extractKeywords(userQuestion);

        const query = `
            SELECT 
                id,
                question,
                answer,
                keywords,
                1 - (embedding <=> $1::vector) as similarity
            FROM semantic_cache
            WHERE 1 - (embedding <=> $1::vector) > $2
            ORDER BY similarity DESC
            LIMIT 5;
        `;

        const { rows } = await pool.query(query, [
            vectorString,
            SIMILARITY_THRESHOLD - 0.05
        ]);

        if (rows.length === 0) {
            return null;
        }

        let bestMatch = null;
        let bestScore = 0;

        for (const row of rows) {
            const cachedKeywords = row.keywords || [];

            const matchCount = userKeywords.filter(k =>
                cachedKeywords.some(ck =>
                    ck.includes(k) || k.includes(ck)
                )
            ).length;

            const keywordScore = matchCount / Math.max(userKeywords.length, 1);
            const combinedScore = (row.similarity * 0.7) + (keywordScore * 0.3);

            if (combinedScore > bestScore && keywordScore >= KEYWORD_MATCH_THRESHOLD) {
                bestScore = combinedScore;
                bestMatch = row;
            }
        }

        if (bestMatch && bestScore > 0.75) {
            return bestMatch.answer;
        }

        return null;

    } catch (error) {
        console.error('Cache lookup failed:', error.message);
        return null;
    }
}

async function saveToSemanticCache(question, answer, queryVector) {
    try {
        if (!queryVector || !Array.isArray(queryVector) || queryVector.length === 0) {
            return;
        }

        if (!question || typeof question !== 'string' || !answer || typeof answer !== 'string') {
            return;
        }

        const normalizedQuestion = normalizeForMatching(question);
        const keywords = extractKeywords(question);
        const vectorString = `[${queryVector.join(',')}]`;

        const checkQuery = `
            SELECT id FROM semantic_cache 
            WHERE question = $1 
            LIMIT 1;
        `;
        const { rows: existingRows } = await pool.query(checkQuery, [normalizedQuestion]);

        if (existingRows.length > 0) {
            await pool.query(
                `UPDATE semantic_cache 
                 SET answer = $1, embedding = $2::vector, keywords = $3, created_at = NOW()
                 WHERE question = $4`,
                [answer, vectorString, keywords, normalizedQuestion]
            );
        } else {
            await pool.query(
                `INSERT INTO semantic_cache (question, answer, embedding, keywords) 
                 VALUES ($1, $2, $3::vector, $4)`,
                [normalizedQuestion, answer, vectorString, keywords]
            );
        }

    } catch (error) {
        console.error('Cache save failed:', error.message);
    }
}

async function cleanOldCache(daysOld = 30) {
    try {
        const result = await pool.query(
            `DELETE FROM semantic_cache 
             WHERE created_at < NOW() - INTERVAL '${daysOld} days'
             RETURNING id`
        );

        return result.rowCount;
    } catch (error) {
        console.error('Cache cleanup failed:', error.message);
        return 0;
    }
}

module.exports = {
    findInSemanticCache,
    saveToSemanticCache,
    cleanOldCache
};