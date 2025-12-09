const { pool } = require('./db');

const SIMILARITY_THRESHOLD = 0.94; // Giảm từ 0.95 xuống 0.92 để dễ hit cache hơn

// ============================================================================
// LOGGER
// ============================================================================
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

// Helper: Loại bỏ HTML và dữ liệu quá dài khỏi logs
function sanitizeMeta(meta) {
    const sanitized = {};
    for (const [key, value] of Object.entries(meta)) {
        if (typeof value === 'string') {
            // Nếu là HTML, chỉ log metadata
            if (value.includes('<!DOCTYPE') || value.includes('<html')) {
                sanitized[key] = '[HTML Content - ' + value.length + ' chars]';
            }
            // Nếu quá dài (>500 chars), cắt bớt
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

// ============================================================================
// Tìm câu trả lời trong cache
// ============================================================================
async function findInSemanticCache(userQuestion, queryVector) {
    try {
        // Validate input
        if (!queryVector || !Array.isArray(queryVector) || queryVector.length === 0) {
            logger.warn('Invalid queryVector for cache lookup', {
                type: typeof queryVector,
                isArray: Array.isArray(queryVector),
                length: queryVector?.length
            });
            return null;
        }

        if (!userQuestion || typeof userQuestion !== 'string') {
            logger.warn('Invalid userQuestion for cache lookup', {
                type: typeof userQuestion
            });
            return null;
        }

        // Chuyển array thành string vector format của PostgreSQL
        const vectorString = `[${queryVector.join(',')}]`;

        // Query tìm kiếm kết hợp:
        // 1. Vector similarity (embedding <=> $1)
        // 2. Text search matching (to_tsvector)
        const query = `
            SELECT 
                answer, 
                1 - (embedding <=> $1::vector) as similarity
            FROM semantic_cache
            WHERE 1 - (embedding <=> $1::vector) > $2
            AND to_tsvector('simple', question) @@ plainto_tsquery('simple', $3)
            ORDER BY similarity DESC
            LIMIT 1;
        `;

        const startTime = Date.now();
        const { rows } = await pool.query(query, [
            vectorString,
            SIMILARITY_THRESHOLD,
            userQuestion
        ]);
        const duration = Date.now() - startTime;

        if (rows.length > 0) {
            const similarity = rows[0].similarity;
            logger.info('Cache HIT', {
                similarity: `${(similarity * 100).toFixed(1)}%`,
                question: userQuestion.substring(0, 80),
                queryTime: `${duration}ms`
            });
            return rows[0].answer;
        }

        logger.info('Cache MISS', {
            question: userQuestion.substring(0, 80),
            reason: 'No similar question found or keyword mismatch',
            queryTime: `${duration}ms`,
            threshold: SIMILARITY_THRESHOLD
        });
        return null;

    } catch (error) {
        logger.error('Cache lookup failed', error);
        return null;
    }
}

// ============================================================================
// Lưu câu trả lời vào cache
// ============================================================================
async function saveToSemanticCache(question, answer, queryVector) {
    try {
        // Validate input
        if (!queryVector || !Array.isArray(queryVector) || queryVector.length === 0) {
            logger.warn('Invalid queryVector for cache save', {
                type: typeof queryVector,
                isArray: Array.isArray(queryVector),
                length: queryVector?.length
            });
            return;
        }

        if (!question || typeof question !== 'string') {
            logger.warn('Invalid question for cache save', {
                type: typeof question
            });
            return;
        }

        if (!answer || typeof answer !== 'string') {
            logger.warn('Invalid answer for cache save', {
                type: typeof answer
            });
            return;
        }

        // Giới hạn độ dài question để tránh duplicate
        const normalizedQuestion = question.trim().substring(0, 500);

        const vectorString = `[${queryVector.join(',')}]`;

        const startTime = Date.now();

        // Kiểm tra xem câu hỏi đã tồn tại chưa
        const checkQuery = `
            SELECT id FROM semantic_cache 
            WHERE question = $1 
            LIMIT 1;
        `;
        const { rows: existingRows } = await pool.query(checkQuery, [normalizedQuestion]);

        if (existingRows.length > 0) {
            // Update câu trả lời cũ
            await pool.query(
                `UPDATE semantic_cache 
                 SET answer = $1, embedding = $2::vector, created_at = NOW()
                 WHERE question = $3`,
                [answer, vectorString, normalizedQuestion]
            );

            const duration = Date.now() - startTime;
            logger.info('Cache updated', {
                question: normalizedQuestion.substring(0, 80),
                answerLength: answer.length,
                vectorLength: queryVector.length,
                duration: `${duration}ms`
            });
        } else {
            // Insert mới
            await pool.query(
                `INSERT INTO semantic_cache (question, answer, embedding) 
                 VALUES ($1, $2, $3::vector)`,
                [normalizedQuestion, answer, vectorString]
            );

            const duration = Date.now() - startTime;
            logger.info('Cache saved (new)', {
                question: normalizedQuestion.substring(0, 80),
                answerLength: answer.length,
                vectorLength: queryVector.length,
                duration: `${duration}ms`
            });
        }

    } catch (error) {
        logger.error('Cache save failed', error);
    }
}

// ============================================================================
// Xóa cache cũ (Optional - chạy định kỳ)
// ============================================================================
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