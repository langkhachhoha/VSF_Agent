"""
Memory Agent with Long-term and Buffer Memory
"""

import os
import logging
from typing import Optional

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from qdrant_client import QdrantClient
from protonx import ProtonX
from openai import OpenAI

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    PROTONX_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_LONGTERM_FILE,
    OPENAI_API_KEY,
)
from tools import (
    RetrieveLongTermMemoryTool,
    SaveMemoryTool,
    RetrieveDoctorTool,
    RetrieveQdrantLongTermTool,
)

# Setup logger
logger = logging.getLogger(__name__)


# ============================================================================
# Memory Agent
# ============================================================================

class MemoryAgent:
    """
    Agent với buffer memory (10 context) và long-term memory
    Thiết kế giống n8n
    """
    
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = DEFAULT_MODEL,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        longterm_file: str = DEFAULT_LONGTERM_FILE
    ):
        """
        Khởi tạo Memory Agent
        
        Args:
            openai_api_key: OpenAI API key
            model_name: Tên model OpenAI (mặc định: gpt-4o-mini)
            buffer_size: Số lượng context lưu trong buffer memory (mặc định: 10)
            longterm_file: Đường dẫn file lưu long-term memory (mặc định: longterm.txt)
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.buffer_size = buffer_size
        self.longterm_file = longterm_file
        
        # Tracking cho priming
        self.message_count_since_prime = 0
        self.is_primed = False
        self.priming_message = "Bạn biết những thông tin cá nhân gì về tôi? Hãy tóm tắt ngắn gọn tất cả thông tin long-term bạn có về tôi."
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        # Khởi tạo OpenAI client trực tiếp
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        self.memory = ConversationBufferWindowMemory(
            k=buffer_size,
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            self.protonx_client = ProtonX(api_key=PROTONX_API_KEY)
            logger.info("✅ Connected to Qdrant and ProtonX")
        except Exception as e:
            logger.error(f"⚠️ Không thể kết nối Qdrant/ProtonX: {str(e)}")
            self.qdrant_client = None
            self.protonx_client = None
        
        self.tools = [
            RetrieveLongTermMemoryTool(
                llm=self.llm,
                openai_client=self.openai_client,
                model_name=model_name,
                longterm_file=longterm_file
            ),
            SaveMemoryTool(
                longterm_file=longterm_file,
                qdrant_client=self.qdrant_client,
                protonx_client=self.protonx_client
            ),
            RetrieveQdrantLongTermTool(
                qdrant_client=self.qdrant_client,
                protonx_client=self.protonx_client
            ),
            RetrieveDoctorTool(
                qdrant_client=self.qdrant_client,
                protonx_client=self.protonx_client
            )
        ]
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
Bạn là một trợ lý AI thông minh, có khả năng tư vấn sức khỏe cơ bản và hỗ trợ người dùng tìm bác sĩ phù hợp. Đồng thời, bạn quản lý và sử dụng long-term memory một cách tối ưu để cá nhân hóa trải nghiệm.

---

## **✨ QUY TRÌNH HOẠT ĐỘNG TỐI ƯU**

### **1. Long-Term Memory (Linh hoạt – chỉ dùng khi cần)**

* Nếu **chưa có** thông tin long-term trong phiên hiện tại → **Gọi `retrieve_long_term_memory`** để tải dữ liệu nền.
* Nếu **đã có** thông tin long-term trong context → **Sử dụng lại**, KHÔNG gọi lại tool.
* Luôn tận dụng thông tin đã biết để cá nhân hóa trả lời (gọi tên, nhắc lại tình trạng sức khỏe, thói quen…).

---

### **2. Semantic Search qua Qdrant (Fallback khi thiếu thông tin)**

* Nếu dữ liệu long-term đang có **không đủ** để trả lời câu hỏi cụ thể của user:
  → **Gọi `retrieve_qdrant_longterm`** để tìm kiếm semantic trong toàn bộ lịch sử.
* Tool trả về **3 thông tin liên quan nhất** (top_k = 3).
* Ví dụ: User hỏi “Tôi từng nói gì về con tôi?” → tool này sẽ tìm lại thông tin cũ.

---

### **3. Tìm Bác Sĩ**

* Khi người dùng cần tư vấn hoặc tìm bác sĩ theo mô tả triệu chứng/chuyên khoa →
  → **Gọi `retrieve_doctor`** với mô tả bệnh hoặc chuyên khoa.
* Trả về danh sách bác sĩ phù hợp nhất.

---

### **4. Lưu Thông Tin Mới (Dual-write)**

* Khi user đưa thêm thông tin cá nhân quan trọng, chưa có trong long-term memory → **Gọi `save_memory`**.
* Hệ thống tự động lưu vào:

  * File long-term memory
  * Database Qdrant (embedding hóa)

---

---

## **✨ CÁC CÔNG CỤ ĐƯỢC HỖ TRỢ**

### **• retrieve_long_term_memory(query)**

* Đọc toàn bộ long-term memory từ file.
* Chỉ dùng khi **context chưa có memory nền**.
* Input: câu hỏi gốc của user.
* Output: toàn bộ dữ liệu long-term đã lưu.

---

### **• retrieve_qdrant_longterm(query, top_k=3)**

* Semantic search trong database long-term memory.
* Dùng khi **memory hiện có không đủ** để trả lời.
* Output: top 3 kết quả liên quan nhất.

---

### **• save_memory(information)**

* Lưu thông tin quan trọng (file + database).
* Thông tin phải được tóm tắt ngắn gọn, chuẩn hoá.
* Ví dụ:

  * “Người dùng tên Minh, 45 tuổi.”
  * “Người dùng bị tiểu đường type 2.”

---

### **• retrieve_doctor(query, top_k=3)**

* Tìm bác sĩ phù hợp theo triệu chứng hoặc chuyên khoa.
* Ví dụ: “bác sĩ nội tiết tiểu đường”, “bác sĩ tim mạch”.

---

---

## ** NGUYÊN TẮC HOẠT ĐỘNG**

* Ưu tiên **giảm số lần gọi tool** (chỉ gọi khi cần thiết).
* Sử dụng lại thông tin đã có trong context tối đa có thể.
* Chỉ dùng `retrieve_qdrant_longterm` khi cần truy vấn chi tiết theo ngữ nghĩa.
* Trả lời tự nhiên, thân thiện, **không nhắc đến việc bạn đang sử dụng tool**.
* Chỉ dùng tool để hỗ trợ, không lạm dụng.

---
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        logger.info("✅ MemoryAgent initialized successfully")
    
    def _prime_memory(self) -> Optional[str]:
        """
        Mồi buffer memory với thông tin long-term
        Gọi agent để tóm tắt thông tin long-term và lưu vào buffer
        
        Returns:
            Tóm tắt thông tin long-term hoặc None nếu không có
        """
        try:
            logger.info("🔄 Đang mồi buffer memory với long-term data...")
            
            # Gọi agent với câu hỏi mồi
            response = self.agent_executor.invoke({"input": self.priming_message})
            summary = response["output"]
            
            # Reset counter
            self.message_count_since_prime = 0
            self.is_primed = True
            
            logger.info(f"✅ Đã mồi buffer memory: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error in priming: {str(e)}")
            return None
    
    def _should_reprime(self) -> bool:
        """
        Kiểm tra xem có cần mồi lại không
        Cần mồi lại khi số message từ lần mồi cuối >= buffer_size
        
        Returns:
            True nếu cần mồi lại
        """
        return self.message_count_since_prime >= self.buffer_size
    
    def chat(self, message: str, auto_prime: bool = True) -> str:
        """
        Gửi tin nhắn đến agent và nhận phản hồi
        
        Args:
            message: Tin nhắn từ người dùng
            auto_prime: Tự động mồi buffer memory nếu cần (mặc định: True)
            
        Returns:
            Phản hồi từ agent
        """
        try:
            # Kiểm tra và mồi nếu cần
            if auto_prime:
                # Mồi lần đầu nếu chưa mồi
                if not self.is_primed:
                    logger.info("🔄 Lần đầu chat, đang mồi buffer memory...")
                    self._prime_memory()
                # Mồi lại nếu đã đến buffer_size
                elif self._should_reprime():
                    logger.info("🔄 Buffer memory sắp đầy, đang mồi lại...")
                    self._prime_memory()
            
            # Chat bình thường
            response = self.agent_executor.invoke({"input": message})
            
            # Tăng counter (chỉ cho user message, không tính priming message)
            self.message_count_since_prime += 1
            
            logger.info(f"📊 Priming status: {self.message_count_since_prime}/{self.buffer_size} messages")
            
            return response["output"]
        except Exception as e:
            logger.error(f"❌ Error in chat: {str(e)}")
            return f"Lỗi: {str(e)}"
    
    def clear_buffer_memory(self):
        """Xóa buffer memory và reset priming state"""
        self.memory.clear()
        self.message_count_since_prime = 0
        self.is_primed = False
        logger.info("✅ Buffer memory cleared và reset priming state")
    
    def view_buffer_memory(self):
        """Xem nội dung buffer memory"""
        messages = self.memory.load_memory_variables({})
        return messages.get("chat_history", [])
    
    def view_longterm_memory(self):
        """Xem nội dung long-term memory"""
        if not os.path.exists(self.longterm_file):
            return "Long-term memory trống"
        
        with open(self.longterm_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content if content.strip() else "Long-term memory trống"
    
    def clear_longterm_memory(self):
        """Xóa long-term memory"""
        if os.path.exists(self.longterm_file):
            os.remove(self.longterm_file)
        logger.info("✅ Long-term memory cleared")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function for testing the agent"""
        
    agent = MemoryAgent(
        openai_api_key=OPENAI_API_KEY,
        model_name=DEFAULT_MODEL,
        buffer_size=DEFAULT_BUFFER_SIZE,
        longterm_file=DEFAULT_LONGTERM_FILE
    )
    
    print("💬 Memory Agent")
    print("Lệnh: /clear_buffer | /clear_longterm | /view_buffer | /view_longterm | /quit\n")
    
    while True:
        try:
            user_input = input("👤 Bạn: ").strip()
            
            if not user_input:
                continue
            
            # Xử lý lệnh đặc biệt
            if user_input.lower() in ['/quit', '/exit']:
                print("👋 Tạm biệt!")
                break
            
            elif user_input.lower() == '/clear_buffer':
                agent.clear_buffer_memory()
                continue
            
            elif user_input.lower() == '/clear_longterm':
                agent.clear_longterm_memory()
                continue
            
            elif user_input.lower() == '/view_buffer':
                messages = agent.view_buffer_memory()
                print("\n📝 Buffer Memory:")
                for msg in messages:
                    role = "User" if msg.type == "human" else "AI"
                    print(f"{role}: {msg.content}")
                print()
                continue
            
            elif user_input.lower() == '/view_longterm':
                content = agent.view_longterm_memory()
                print(f"\n💾 Long-term Memory:\n{content}\n")
                continue
            
            # Chat bình thường
            response = agent.chat(user_input)
            print(f"\n🤖 {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {str(e)}\n")


if __name__ == "__main__":
    main()
