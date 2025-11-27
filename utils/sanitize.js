function sanitizeQuestion(question) {
  if (!question || typeof question !== 'string') return "";
  // Xóa khoảng trắng thừa, cắt ngắn bớt nếu quá dài
  return question.trim().replace(/\s+/g, ' ').substring(0, 1000);
}
module.exports = { sanitizeQuestion };