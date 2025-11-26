"""
Streamlit Frontend cho Memory Agent
Giao diện giống ChatGPT với sidebar hiển thị Personalize Data
"""

import streamlit as st
import requests
from typing import List, Dict
import time
import pandas as pd
import re

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://localhost:8000"

# ============================================================================
# Page Config
# ============================================================================

st.set_page_config(
    page_title="Memory Agent - Doctor Vinmec",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS - ChatGPT Style
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #343541;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ececf1;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #444654;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* User message */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: #343541;
    }
    
    /* Assistant message */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background-color: #444654;
    }
    
    /* Input box */
    .stChatInputContainer {
        background-color: #40414f;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton button:hover {
        background-color: #0d8c6c;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2e3a;
        color: #ececf1;
        border-radius: 6px;
    }
    
    /* Text */
    h1, h2, h3, p, label {
        color: #ececf1 !important;
    }
    
    /* Memory box */
    .memory-box {
        background-color: #2d2e3a;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ececf1;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2d2e3a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #777;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API Functions
# ============================================================================

def chat_with_agent(message: str) -> str:
    """Gửi message đến agent và nhận response"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message, "session_id": "default"},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"❌ Lỗi kết nối API: {str(e)}"
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

def get_longterm_memory() -> str:
    """Lấy nội dung long-term memory"""
    try:
        response = requests.get(f"{API_BASE_URL}/memory/longterm", timeout=10)
        response.raise_for_status()
        return response.json()["content"]
    except requests.exceptions.RequestException as e:
        return f"❌ Lỗi kết nối API: {str(e)}"
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

def parse_longterm_memory(content: str) -> pd.DataFrame:
    """Parse long-term memory thành DataFrame với cột Time và Information"""
    if not content or content == "Long-term memory trống" or "❌" in content:
        return pd.DataFrame(columns=["Time", "Information"])
    
    lines = content.strip().split('\n')
    data = []
    
    for line in lines:
        # Parse format: [YYYY-MM-DD HH:MM:SS] Information
        match = re.match(r'\[([\d\-\s:]+)\]\s*(.*)', line)
        if match:
            time_str = match.group(1)
            info = match.group(2)
            data.append({"Time": time_str, "Information": info})
    
    return pd.DataFrame(data)

def clear_longterm_memory() -> bool:
    """Xóa long-term memory"""
    try:
        response = requests.delete(f"{API_BASE_URL}/memory/longterm", timeout=10)
        response.raise_for_status()
        return True
    except:
        return False

def clear_buffer_memory() -> bool:
    """Xóa buffer memory"""
    try:
        response = requests.delete(f"{API_BASE_URL}/memory/buffer", timeout=10)
        response.raise_for_status()
        return True
    except:
        return False

def check_api_health() -> bool:
    """Kiểm tra API server có hoạt động không"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_buffer_memory() -> List[Dict]:
    """Lấy buffer memory"""
    try:
        response = requests.get(f"{API_BASE_URL}/memory/buffer", timeout=10)
        response.raise_for_status()
        return response.json()["messages"]
    except:
        return []

def get_tool_history() -> Dict:
    """Lấy lịch sử tool calls"""
    try:
        response = requests.get(f"{API_BASE_URL}/tools/history", timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return {"total_calls": 0, "history": []}

def clear_tool_history() -> bool:
    """Xóa lịch sử tool calls"""
    try:
        response = requests.delete(f"{API_BASE_URL}/tools/history", timeout=10)
        response.raise_for_status()
        return True
    except:
        return False

# ============================================================================
# Session State
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_connected" not in st.session_state:
    st.session_state.api_connected = check_api_health()

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("⚙️ Memory Agent")
    st.markdown("---")
    
    # Priming Status
    try:
        response = requests.get(f"{API_BASE_URL}/priming/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            st.markdown("### 🔄 Priming Status")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Is Primed", "✅" if status["is_primed"] else "❌")
            with col2:
                st.metric("Messages", f"{status['message_count_since_prime']}/{status['buffer_size']}")
            
            if status["should_reprime"]:
                st.warning("⚠️ Sẽ re-prime ở message tiếp theo")
            st.markdown("---")
    except:
        pass
    
    # New Chat button
    if st.button("🆕 New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### 📊 Memory Management")
    
    # View Long-term Memory
    if st.button("👁️ View Long-term Memory", use_container_width=True):
        with st.spinner("Đang tải..."):
            content = get_longterm_memory()
            if content and "❌" not in content:
                df = parse_longterm_memory(content)
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Long-term memory trống")
            else:
                st.error(content)
    
    # View Buffer Memory
    if st.button("💭 View Buffer Memory", use_container_width=True):
        with st.spinner("Đang tải..."):
            messages = get_buffer_memory()
            if messages:
                for msg in messages:
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    st.text(f"{role_icon} {msg['role'].upper()}")
                    st.text(msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"])
                    st.markdown("---")
            else:
                st.info("Buffer memory trống")
    
    # View Tool History
    if st.button("🔧 View Tool History", use_container_width=True):
        with st.spinner("Đang tải..."):
            history = get_tool_history()
            st.metric("Total Tool Calls", history["total_calls"])
            if history["history"]:
                for i, call in enumerate(history["history"][-5:]):  # Show last 5
                    with st.expander(f"Call #{len(history['history']) - len(history['history'][-5:]) + i + 1}: {call['tool_name']}"):
                        st.text(f"Time: {call['timestamp']}")
                        st.text(f"User: {call['user_message'][:50]}...")
                        st.text(f"Output: {call['tool_output'][:100]}...")
            else:
                st.info("Chưa có tool calls")
    
    st.markdown("### 🗑️ Clear Memory")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clear Buffer", use_container_width=True):
            if clear_buffer_memory():
                st.success("✅ Đã xóa buffer!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Lỗi xóa buffer")
    
    with col2:
        if st.button("🗑️ Clear Long-term", use_container_width=True):
            if clear_longterm_memory():
                st.success("✅ Đã xóa long-term!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Lỗi xóa long-term")
    
    if st.button("🧽 Clear Tool History", use_container_width=True):
        if clear_tool_history():
            st.success("✅ Đã xóa tool history!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Lỗi xóa tool history")

# ============================================================================
# Main Chat Interface
# ============================================================================

st.title("💬 Memory Agent - Doctor Vinmec")
st.caption("Trợ lý AI thông minh với khả năng ghi nhớ và tìm kiếm bác sĩ")

# Check API connection
if not st.session_state.api_connected:
    st.error("⚠️ Không thể kết nối đến API server. Vui lòng khởi động API server trước.")
    st.code("python api_server.py", language="bash")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Nhập tin nhắn của bạn..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            response = chat_with_agent(prompt)
        st.markdown(response)
    
    # Add assistant message to chat
    st.session_state.messages.append({"role": "assistant", "content": response})

# Welcome message
if len(st.session_state.messages) == 0:
    st.info("""
    👋 **Chào mừng bạn đến với Memory Agent!**
    
    Tôi có thể giúp bạn:
    - 💬 Trò chuyện và ghi nhớ thông tin về bạn
    - 👨‍⚕️ Tìm kiếm bác sĩ phù hợp với tình trạng sức khỏe
    - 🎯 Cá nhân hóa câu trả lời dựa trên thông tin đã lưu
    
    Hãy bắt đầu bằng cách giới thiệu về bản thân hoặc hỏi tôi bất cứ điều gì!
    """)

# Footer
st.markdown("---")
st.caption("Powered by OpenAI GPT-4o-mini | Built with FastAPI & Streamlit")

