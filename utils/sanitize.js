function sanitizeQuestion(question) {
  if (!question || typeof question !== 'string') return "";

  let cleaned = question.trim().replace(/\s+/g, ' ');
  cleaned = cleaned.replace(/[<>{}[\]\\]/g, '');

  return cleaned.substring(0, 1000);
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
  extractKeywords
};