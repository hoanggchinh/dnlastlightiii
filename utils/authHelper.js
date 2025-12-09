const bcrypt = require('bcryptjs');

async function hashPassword(plainPassword) {
    const salt = await bcrypt.genSalt(10);
    return await bcrypt.hash(plainPassword, salt);
}

async function comparePassword(plainPassword, dbHash) {
    return await bcrypt.compare(plainPassword, dbHash);
}

function generateOTP() {
    return Math.floor(100000 + Math.random() * 900000).toString();
}

module.exports = { hashPassword, comparePassword, generateOTP };