function sanitizeQuestion(question) {
  if (!question || typeof question !== 'string') return "";
  return question.trim().replace(/\s+/g, ' ').substring(0, 1000);
}
module.exports = { sanitizeQuestion };