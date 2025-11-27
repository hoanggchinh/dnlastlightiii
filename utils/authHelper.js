const bcrypt = require('bcryptjs');

// Mã hóa mật khẩu
async function hashPassword(plainPassword) {
    const salt = await bcrypt.genSalt(10);
    return await bcrypt.hash(plainPassword, salt);
}

// Kiểm tra mật khẩu
async function comparePassword(plainPassword, dbHash) {
    return await bcrypt.compare(plainPassword, dbHash);
}

// Tạo OTP 6 số ngẫu nhiên
function generateOTP() {
    return Math.floor(100000 + Math.random() * 900000).toString();
}

module.exports = { hashPassword, comparePassword, generateOTP };