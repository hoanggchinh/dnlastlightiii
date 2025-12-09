const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const nodemailer = require('nodemailer');
require('dotenv').config();

const askHandler = require('./api/ask');
const { pool } = require('./utils/db');
const { hashPassword, comparePassword, generateOTP } = require('./utils/authHelper');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json());

const transporter = nodemailer.createTransport({
    host: "smtp.gmail.com",
    port: 465,
    secure: true,
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
    }
});

transporter.verify((error, success) => {
    if (error) {
        console.log('KẾT NỐI EMAIL THẤT BẠI:', error);
    } else {
        console.log('Server email đã kết nối thành công với: ' + process.env.EMAIL_USER);
    }
});

app.post('/api/ask', askHandler);

app.post('/api/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        const result = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
        const user = result.rows[0];

        if (!user) return res.status(400).json({ success: false, message: "Email chưa đăng ký" });
        if (!user.is_verified) return res.status(400).json({ success: false, message: "Tài khoản chưa xác thực OTP" });

        if (!user.password_hash) return res.status(400).json({ success: false, message: "Lỗi dữ liệu tài khoản" });

        const isMatch = await comparePassword(password, user.password_hash);
        if (!isMatch) return res.status(400).json({ success: false, message: "Sai mật khẩu" });

        res.json({ success: true, userId: user.id, user: { name: user.email }, message: "Đăng nhập thành công" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: "Lỗi Server" });
    }
});


app.post('/api/send-otp', async (req, res) => {
    try {
        const { email, type } = req.body;
        const otp = generateOTP();
        const expiresAt = new Date(Date.now() + 5 * 60 * 1000);

        const userCheck = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
        const user = userCheck.rows[0];

        if (type === 'register') {

            if (user && user.is_verified) {
                return res.status(400).json({ success: false, message: "Email này đã được sử dụng." });
            }

            if (!user) {
                await pool.query(
                    `INSERT INTO users (email, otp_code, otp_expires_at, is_verified) VALUES ($1, $2, $3, FALSE)`,
                    [email, otp, expiresAt]
                );
            } else {
                await pool.query(
                    `UPDATE users SET otp_code = $1, otp_expires_at = $2 WHERE email = $3`,
                    [otp, expiresAt, email]
                );
            }
        } else if (type === 'forgot') {
            if (!user || !user.is_verified) {
                return res.status(400).json({ success: false, message: "Email không tồn tại trong hệ thống." });
            }
            await pool.query(
                `UPDATE users SET otp_code = $1, otp_expires_at = $2 WHERE email = $3`,
                [otp, expiresAt, email]
            );
        }

        const mailOptions = {
            from: `"Tomtitmui OS Support" <${process.env.EMAIL_USER}>`,
            to: email,
            subject: `Mã xác thực của bạn: ${otp}`,
            text: `Mã OTP của bạn là: ${otp}. Mã này sẽ hết hạn trong 5 phút.`,
            html: `<div style="font-family: Arial, sans-serif; padding: 20px;">
                    <h2>Xin chào!</h2>
                    <p>Bạn đang thực hiện xác thực tài khoản tại Tomtitmui OS.</p>
                    <p>Mã OTP của bạn là:</p>
                    <h1 style="color: #0071e3; letter-spacing: 5px;">${otp}</h1>
                    <p>Mã có hiệu lực trong 5 phút. Vui lòng không chia sẻ mã này cho ai.</p>
                   </div>`
        };

        await transporter.sendMail(mailOptions);

        console.log(`Đã gửi OTP đến: ${email}`);
        res.json({ success: true, message: "Đã gửi mã OTP đến email của bạn." });

    } catch (err) {
    console.error("Lỗi gửi OTP:", err);
    console.error("Chi tiết:", err.message);
    res.status(500).json({
        success: false,
        message: "Không thể gửi email. Vui lòng kiểm tra lại địa chỉ.",
        error: err.message
    });
}
});

app.post('/api/register', async (req, res) => {
    try {
        const { email, password, otp } = req.body;

        const result = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
        const user = result.rows[0];

        if (!user) return res.status(400).json({ success: false, message: "Email không hợp lệ (hãy yêu cầu gửi lại OTP)" });

        if (user.otp_code !== otp) return res.status(400).json({ success: false, message: "Mã OTP không đúng" });
        if (new Date() > new Date(user.otp_expires_at)) return res.status(400).json({ success: false, message: "Mã OTP đã hết hạn" });

        const hashedPassword = await hashPassword(password);

        await pool.query(
            `UPDATE users SET password_hash = $1, is_verified = TRUE, otp_code = NULL WHERE email = $2`,
            [hashedPassword, email]
        );

        res.json({ success: true, message: "Đăng ký thành công!" });

    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: "Lỗi đăng ký" });
    }
});

app.post('/api/reset-password', async (req, res) => {
    try {
        const { email, otp, newPassword } = req.body;

        const result = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
        const user = result.rows[0];

        if (!user) {
            return res.status(400).json({ success: false, message: "Email không tồn tại" });
        }

        if (user.otp_code !== otp) {
            return res.status(400).json({ success: false, message: "Mã OTP không đúng" });
        }
        if (new Date() > new Date(user.otp_expires_at)) {
            return res.status(400).json({ success: false, message: "Mã OTP đã hết hạn" });
        }

        const hashedPassword = await hashPassword(newPassword);

        await pool.query(
            `UPDATE users SET password_hash = $1, otp_code = NULL WHERE email = $2`,
            [hashedPassword, email]
        );

        res.json({ success: true, message: "Đổi mật khẩu thành công!" });

    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: "Lỗi server" });
    }
});

app.get('/api/chats', async (req, res) => {
    try {
        const { userId } = req.query;
        if (!userId) return res.json([]);
        const result = await pool.query('SELECT * FROM chats WHERE user_id = $1 ORDER BY created_at DESC', [userId]);
        res.json(result.rows);
    } catch (err) { res.status(500).json([]); }
});

app.get('/api/messages', async (req, res) => {
    try {
        const { chatId } = req.query;
        if (!chatId) return res.json([]);
        const result = await pool.query('SELECT * FROM messages WHERE chat_id = $1 ORDER BY created_at ASC', [chatId]);
        res.json(result.rows);
    } catch (err) { res.status(500).json([]); }
});

if (require.main === module) {
    app.listen(PORT, () => {
        console.log(`Server API đang chạy tại http://localhost:${PORT}`);
    });
}
module.exports = app;