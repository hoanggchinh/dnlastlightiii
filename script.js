document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    // Hàm cuộn xuống tin nhắn cuối cùng
    const scrollToBottom = () => {
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    // Hàm hiển thị tin nhắn
    const displayMessage = (text, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message');
        msgDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
        msgDiv.innerHTML = text; // Dùng innerHTML để hỗ trợ hiển thị Markdown đơn giản
        chatBox.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    };

    // Hàm gửi tin nhắn
    const sendMessage = async () => {
        const question = chatInput.value.trim();
        if (question === '') return;

        // 1. Hiển thị tin nhắn người dùng
        displayMessage(question, 'user');
        chatInput.value = '';
        sendBtn.disabled = true; // Vô hiệu hóa nút Gửi

        // 2. Hiển thị trạng thái đang gõ
        const typingIndicator = displayMessage('Bot đang gõ...', 'bot');
        typingIndicator.classList.add('typing-indicator');

        try {
            // 3. Gửi câu hỏi đến Vercel Serverless Function
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // 4. Xóa trạng thái đang gõ
            chatBox.removeChild(typingIndicator);

            // 5. Hiển thị câu trả lời của Bot
            let botAnswer = data.answer || "Xin lỗi, đã có lỗi xảy ra hoặc câu trả lời rỗng.";
            displayMessage(botAnswer, 'bot');

        } catch (error) {
            console.error('Lỗi khi gửi câu hỏi:', error);
            // 4. Xóa trạng thái đang gõ
            chatBox.removeChild(typingIndicator);
            // 5. Hiển thị lỗi
            displayMessage('Xin lỗi, đã xảy ra lỗi kết nối hoặc lỗi server. Vui lòng kiểm tra lại console.', 'bot');
        } finally {
            sendBtn.disabled = false; // Kích hoạt lại nút Gửi
        }
    };

    // Bắt sự kiện Gửi khi nhấn nút
    sendBtn.addEventListener('click', sendMessage);

    // Bắt sự kiện Gửi khi nhấn Enter
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});