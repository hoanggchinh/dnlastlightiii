function detectXSS(input) {
  if (!input || typeof input !== 'string') return false;

  const xssPatterns = [
    /<script[\s\S]*?>/i,
    /javascript:/i,
    /on\w+\s*=/i,
    /<iframe/i,
    /<embed/i,
    /<object/i,
    /onerror\s*=/i,
    /onload\s*=/i,
    /<img[^>]+src[^>]*>/i,
    /eval\s*\(/i,
    /expression\s*\(/i,
    /<svg[\s\S]*?onload/i,
    /data:text\/html/i,
    /vbscript:/i
  ];

  return xssPatterns.some(pattern => pattern.test(input));
}

function sanitizeQuestion(question) {
  if (!question || typeof question !== 'string') return "";

  const hasXSS = detectXSS(question);

  let cleaned = question.trim().replace(/\s+/g, ' ');

  cleaned = cleaned
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<\/?[^>]+(>|$)/g, '');

  cleaned = cleaned
    .replace(/javascript:/gi, '')
    .replace(/data:text\/html/gi, '')
    .replace(/on\w+\s*=/gi, '')
    .replace(/vbscript:/gi, '');

  cleaned = cleaned.replace(/[{}[\]\\]/g, '');

  cleaned = cleaned
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');

  return {
    sanitized: cleaned.substring(0, 1000),
    hasXSS
  };
}

function normalizeForMatching(question) {
  if (!question || typeof question !== 'string') return "";

  return question
    .toLowerCase()
    .trim()
    .replace(/hoc phi/gi, 'học phí')
    .replace(/diem ren luyen/gi, 'điểm rèn luyện')
    .replace(/hoc bong/gi, 'học bổng')
    .replace(/tot nghiep/gi, 'tốt nghiệp')
    .replace(/xet tot nghiep/gi, 'xét tốt nghiệp')
    .replace(/[.,!?;:]/g, '')
    .replace(/\s+/g, ' ')
    .substring(0, 500);
}

function extractKeywords(question) {
  if (!question || typeof question !== 'string') return [];

  const stopwords = new Set([
    'là', 'của', 'và', 'có', 'được', 'trong', 'với', 'để', 'cho',
    'các', 'này', 'đó', 'khi', 'như', 'sao', 'gì', 'ai', 'ở', 'thì',
    'bao', 'nhiêu', 'nào', 'đâu', 'thế', 'ra', 'về', 'từ', 'trên',
    'dưới', 'giữa', 'sau', 'trước', 'bên', 'cạnh', 'theo'
  ]);

  return question
    .toLowerCase()
    .split(/\s+/)
    .filter(word =>
      word.length > 2 &&
      !stopwords.has(word) &&
      !/^\d+$/.test(word)
    )
    .slice(0, 15);
}

module.exports = {
  sanitizeQuestion,
  normalizeForMatching,
  extractKeywords,
  detectXSS
};