const { pool } = require('./db');
const { normalizeForMatching, extractKeywords } = require('./sanitize');

const SIMILARITY_THRESHOLD = 0.95;
const KEYWORD_MATCH_THRESHOLD = 0.75;
const CACHE_TTL_MINUTES = 5;

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

        if (userKeywords.length < 2) {
            return null;
        }

        const query = `
            SELECT 
                id,
                question,
                answer,
                keywords,
                1 - (embedding <=> $1::vector) as similarity
            FROM semantic_cache
            WHERE 1 - (embedding <=> $1::vector) > $2
            AND created_at > NOW() - INTERVAL '${CACHE_TTL_MINUTES} minutes'
            ORDER BY similarity DESC
            LIMIT 3;
        `;

        const { rows } = await pool.query(query, [
            vectorString,
            SIMILARITY_THRESHOLD
        ]);

        if (rows.length === 0) {
            return null;
        }

        let bestMatch = null;
        let bestScore = 0;

        for (const row of rows) {
            const cachedKeywords = row.keywords || [];

            let exactMatchCount = 0;
            let partialMatchCount = 0;

            userKeywords.forEach(userWord => {
                const hasExactMatch = cachedKeywords.some(ck =>
                    ck === userWord || userWord === ck
                );
                const hasPartialMatch = cachedKeywords.some(ck =>
                    (ck.includes(userWord) && userWord.length > 3) ||
                    (userWord.includes(ck) && ck.length > 3)
                );

                if (hasExactMatch) {
                    exactMatchCount++;
                } else if (hasPartialMatch) {
                    partialMatchCount++;
                }
            });

            const totalKeywords = Math.max(userKeywords.length, cachedKeywords.length);
            const keywordScore = (exactMatchCount * 1.0 + partialMatchCount * 0.3) / totalKeywords;

            const combinedScore = (row.similarity * 0.6) + (keywordScore * 0.4);

            const requiredMinScore = 0.85;

            if (combinedScore > bestScore &&
                keywordScore >= KEYWORD_MATCH_THRESHOLD &&
                exactMatchCount >= Math.min(2, userKeywords.length) &&
                combinedScore >= requiredMinScore) {
                bestScore = combinedScore;
                bestMatch = row;
            }
        }

        if (bestMatch && bestScore >= 0.85) {
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

async function cleanExpiredCache() {
    try {
        const result = await pool.query(
            `DELETE FROM semantic_cache 
             WHERE created_at < NOW() - INTERVAL '${CACHE_TTL_MINUTES} minutes'
             RETURNING id`
        );

        if (result.rowCount > 0) {
            console.log(`Cleaned ${result.rowCount} expired cache entries`);
        }

        return result.rowCount;
    } catch (error) {
        console.error('Cache cleanup failed:', error.message);
        return 0;
    }
}

function startCacheCleanupJob() {
    const cleanupIntervalMs = CACHE_TTL_MINUTES * 60 * 1000;

    setInterval(async () => {
        await cleanExpiredCache();
    }, cleanupIntervalMs);

    console.log(`Cache cleanup job started: running every ${CACHE_TTL_MINUTES} minutes`);
}

module.exports = {
    findInSemanticCache,
    saveToSemanticCache,
    cleanExpiredCache,
    startCacheCleanupJob
};