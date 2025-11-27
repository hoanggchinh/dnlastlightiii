const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
require('dotenv').config();

// Import các công cụ đã làm
const askHandler = require('./api/ask'); // Logic RAG
const { pool } = require('./utils/db'); // Kết nối DB
const { hashPassword, comparePassword, generateOTP } = require('./utils/authHelper');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
//app.use(express.static('public')); // Nếu bạn để index.html trong thư mục public
// Nếu index.html ở ngoài cùng, dùng dòng dưới:
app.use(express.static(__dirname));

// ==========================================
// 1. API CHATBOT (RAG)
// ==========================================
app.post('/ask', askHandler);

// ==========================================
// 2. API TÀI KHOẢN (AUTH)
// ==========================================

// A. Đăng nhập
app.post('/api/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        // 1. Tìm user trong DB
        const result = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
        const user = result.rows[0];

        if (!user) return res.status(400).json({ success: false, message: "Email không tồn tại" });

        // 2. Kiểm tra mật khẩu
        const isMatch = await comparePassword(password, user.password_hash);
        if (!isMatch) return res.status(400).json({ success: false, message: "Sai mật khẩu" });

        // 3. (Optional) Kiểm tra đã xác thực email chưa
        // if (!user.is_verified) return res.status(400).json({ success: false, message: "Vui lòng xác thực email" });

        res.json({ success: true, userId: user.id, message: "Đăng nhập thành công" });

    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: "Lỗi Server" });
    }
});

// B. Gửi OTP (Giả lập - In ra Console để test cho nhanh)
app.post('/api/send-otp', async (req, res) => {
    try {
        const { email, type } = req.body; // type: 'register' hoặc 'forgot'
        const otp = generateOTP();
        const expiresAt = new Date(Date.now() + 5 * 60 * 1000); // Hết hạn sau 5 phút

        // Kiểm tra xem email đã tồn tại chưa (nếu đăng ký thì không được trùng)
        const userCheck = await pool.query('SELECT * FROM users WHERE email = $1', [email]);

        if (type === 'register' && userCheck.rows.length > 0) {
            // Nếu user đã tồn tại nhưng chưa verify thì cho phép gửi lại OTP
            if (userCheck.rows[0].is_verified) {
                return res.status(400).json({ success: false, message: "Email này đã được sử dụng" });
            }
        }

        // Lưu OTP vào DB (Update nếu user đã có, hoặc Insert tạm nếu chưa logic phức tạp)
        // ĐƠN GIẢN HÓA: Ta lưu OTP vào bảng users.
        // Nếu user chưa có (đang đăng ký), ta tạo user tạm với pass rỗng.
        if (userCheck.rows.length === 0) {
             await pool.query(
                `INSERT INTO users (email, otp_code, otp_expires_at) VALUES ($1, $2, $3)`,
                [email, otp, expiresAt]
            );
        } else {
            await pool.query(
                `UPDATE users SET otp_code = $1, otp_expires_at = $2 WHERE email = $3`,
                [otp, expiresAt, email]
            );
        }

        // --- GỬI EMAIL THẬT (NODEMAILER) ---
        // Phần này cần cấu hình SMTP Gmail. Để test nhanh, ta IN RA CONSOLE:
        console.log(`💌 [MOCK EMAIL] Gửi đến ${email} - Mã OTP là: ${otp}`);

        res.json({ success: true, message: "Đã gửi OTP (Check Console server)" });

    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: "Lỗi gửi OTP" });
    }
});

// C. Xác nhận Đăng ký (Verify OTP & Set Password)
app.post('/api/register', async (req, res) => {
    try {
        const { email, password, otp } = req.body;

        const result = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
        const user = result.rows[0];

        if (!user) return res.status(400).json({ success: false, message: "Vui lòng yêu cầu gửi OTP trước" });

        // Kiểm tra OTP
        if (user.otp_code !== otp) return res.status(400).json({ success: false, message: "Sai mã OTP" });
        if (new Date() > new Date(user.otp_expires_at)) return res.status(400).json({ success: false, message: "OTP hết hạn" });

        // Hash mật khẩu
        const hashedPassword = await hashPassword(password);

        // Cập nhật User chính thức
        await pool.query(
            `UPDATE users SET password_hash = $1, is_verified = TRUE, otp_code = NULL WHERE email = $2`,
            [hashedPassword, email]
        );

        res.json({ success: true, message: "Đăng ký thành công" });

    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: "Lỗi đăng ký" });
    }
});

// ==========================================
// 3. API LỊCH SỬ CHAT (HISTORY)
// ==========================================

// A. Lấy danh sách các đoạn chat
app.get('/api/chats', async (req, res) => {
    try {
        const { userId } = req.query;
        if (!userId) return res.json([]);

        const result = await pool.query(
            `SELECT * FROM chats WHERE user_id = $1 ORDER BY created_at DESC`,
            [userId]
        );
        res.json(result.rows);
    } catch (err) {
        console.error(err);
        res.status(500).json([]);
    }
});

// B. Lấy nội dung tin nhắn của 1 đoạn chat
app.get('/api/messages', async (req, res) => {
    try {
        const { chatId } = req.query;
        if (!chatId) return res.json([]);

        const result = await pool.query(
            `SELECT * FROM messages WHERE chat_id = $1 ORDER BY created_at ASC`,
            [chatId]
        );
        res.json(result.rows);
    } catch (err) {
        console.error(err);
        res.status(500).json([]);
    }
});

// Chạy Server
app.listen(PORT, () => {
    console.log(`🚀 Server đang chạy tại http://localhost:${PORT}`);
    console.log(`👉 Mở trình duyệt và test thử chức năng Login/Chat`);
});