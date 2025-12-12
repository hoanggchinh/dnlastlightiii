const { pool } = require('./db');
const { normalizeForMatching, extractKeywords } = require('./sanitize');

const SIMILARITY_THRESHOLD = 0.92;
const KEYWORD_MATCH_THRESHOLD = 0.6;

const logger = {
    info: (message, meta = {}) => {
        console.log(JSON.stringify({
            level: 'info',
            message,
            timestamp: new Date().toISOString(),
            ...sanitizeMeta(meta)
        }));
    },
    warn: (message, meta = {}) => {
        console.warn(JSON.stringify({
            level: 'warn',
            message,
            timestamp: new Date().toISOString(),
            ...sanitizeMeta(meta)
        }));
    },
    error: (message, error = null) => {
        const errorInfo = error ? {
            message: error.message,
            name: error.name,
            code: error.code,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        } : undefined;

        console.error(JSON.stringify({
            level: 'error',
            message,
            timestamp: new Date().toISOString(),
            error: errorInfo
        }));
    }
};

function sanitizeMeta(meta) {
    const sanitized = {};
    for (const [key, value] of Object.entries(meta)) {
        if (typeof value === 'string') {
            if (value.includes('<!DOCTYPE') || value.includes('<html')) {
                sanitized[key] = '[HTML Content - ' + value.length + ' chars]';
            }
            else if (value.length > 500) {
                sanitized[key] = value.substring(0, 500) + '... [truncated]';
            }
            else {
                sanitized[key] = value;
            }
        } else {
            sanitized[key] = value;
        }
    }
    return sanitized;
}

async function findInSemanticCache(userQuestion, queryVector) {
    try {
        if (!queryVector || !Array.isArray(queryVector) || queryVector.length === 0) {
            logger.warn('Invalid queryVector for cache lookup');
            return null;
        }

        if (!userQuestion || typeof userQuestion !== 'string') {
            logger.warn('Invalid userQuestion for cache lookup');
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

        const startTime = Date.now();
        const { rows } = await pool.query(query, [
            vectorString,
            SIMILARITY_THRESHOLD - 0.05
        ]);
        const duration = Date.now() - startTime;

        if (rows.length === 0) {
            logger.info('Cache MISS', {
                question: userQuestion.substring(0, 80),
                reason: 'No similar vectors found',
                queryTime: `${duration}ms`
            });
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
            logger.info('Cache HIT', {
                similarity: `${(bestMatch.similarity * 100).toFixed(1)}%`,
                combinedScore: `${(bestScore * 100).toFixed(1)}%`,
                question: userQuestion.substring(0, 80),
                queryTime: `${duration}ms`
            });
            return bestMatch.answer;
        }

        logger.info('Cache MISS', {
            question: userQuestion.substring(0, 80),
            reason: 'Keyword match too low',
            bestScore: `${(bestScore * 100).toFixed(1)}%`,
            queryTime: `${duration}ms`
        });
        return null;

    } catch (error) {
        logger.error('Cache lookup failed', error);
        return null;
    }
}

async function saveToSemanticCache(question, answer, queryVector) {
    try {
        if (!queryVector || !Array.isArray(queryVector) || queryVector.length === 0) {
            logger.warn('Invalid queryVector for cache save');
            return;
        }

        if (!question || typeof question !== 'string' || !answer || typeof answer !== 'string') {
            logger.warn('Invalid question or answer for cache save');
            return;
        }

        const normalizedQuestion = normalizeForMatching(question);
        const keywords = extractKeywords(question);
        const vectorString = `[${queryVector.join(',')}]`;

        const startTime = Date.now();

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

            const duration = Date.now() - startTime;
            logger.info('Cache updated', {
                question: normalizedQuestion.substring(0, 80),
                duration: `${duration}ms`
            });
        } else {
            await pool.query(
                `INSERT INTO semantic_cache (question, answer, embedding, keywords) 
                 VALUES ($1, $2, $3::vector, $4)`,
                [normalizedQuestion, answer, vectorString, keywords]
            );

            const duration = Date.now() - startTime;
            logger.info('Cache saved', {
                question: normalizedQuestion.substring(0, 80),
                duration: `${duration}ms`
            });
        }

    } catch (error) {
        logger.error('Cache save failed', error);
    }
}

async function cleanOldCache(daysOld = 30) {
    try {
        const result = await pool.query(
            `DELETE FROM semantic_cache 
             WHERE created_at < NOW() - INTERVAL '${daysOld} days'
             RETURNING id`
        );

        logger.info('Old cache cleaned', {
            deletedCount: result.rowCount,
            daysOld
        });

        return result.rowCount;
    } catch (error) {
        logger.error('Cache cleanup failed', error);
        return 0;
    }
}

module.exports = {
    findInSemanticCache,
    saveToSemanticCache,
    cleanOldCache
};