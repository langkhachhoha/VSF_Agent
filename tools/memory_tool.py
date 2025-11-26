"""
Memory Tools: Retrieve and Save Long-term Memory
"""

import os
import logging
from typing import Optional, Type
from datetime import datetime

from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from protonx import ProtonX

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from config import LONGTERM_COLLECTION_NAME

logger = logging.getLogger(__name__)


# ============================================================================
# Tool 1: Retrieve Long Term Memory
# ============================================================================

class RetrieveLongTermMemoryInput(BaseModel):
    """Input schema cho Retrieve Long Term Memory tool"""
    query: str = Field(
        description="Câu hỏi hiện tại để tìm kiếm thông tin liên quan trong long-term memory"
    )


class RetrieveLongTermMemoryTool(BaseTool):
    """Tool để truy vấn thông tin long-term từ file longterm.txt"""
    
    name: str = "retrieve_long_term_memory"
    description: str = """
    LUÔN LUÔN gọi tool này ĐẦU TIÊN trước khi trả lời bất kỳ câu hỏi nào của người dùng.
    Tool này truy vấn thông tin cá nhân và dài hạn đã lưu về người dùng để cá nhân hóa câu trả lời.
    Giống như ChatGPT Personalization - luôn kiểm tra xem có thông tin nào hữu ích không.
    Input: Câu hỏi của người dùng (không sửa đổi)
    Output: Thông tin cá nhân liên quan (nếu có)
    """
    args_schema: Type[BaseModel] = RetrieveLongTermMemoryInput
    longterm_temp_file: str = "longterm_temp.txt"
    longterm_file: str = "longterm.txt"
    llm: Optional[ChatOpenAI] = None
    openai_client: Optional[OpenAI] = None
    model_name: str = "gpt-4o-mini"
    
    def _run(self, query: str) -> str:
        """Thực thi tool để lấy thông tin từ long-term memory"""
        # Get tracer for this tool
        tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)
        
        # Create counter for tool invocations
        tool_counter = meter.create_counter(
            name="tool.invocations",
            description="Number of tool invocations",
            unit="1"
        )
        
        # Start span for this tool execution
        with tracer.start_as_current_span(
            "retrieve_long_term_memory",
            attributes={
                "tool.name": "retrieve_long_term_memory",
                "tool.input.query": query[:500] if query else "",  # Limit to 500 chars
                "tool.file": self.longterm_file
            }
        ) as span:
            try:
                logger.info(f"🔍 Retrieving long-term memory for query: {query[:50]}...")
                tool_counter.add(1, {"tool.name": "retrieve_long_term_memory", "status": "started"})
                if not os.path.exists(self.longterm_file):
                    output = "Không có thông tin long-term memory nào được lưu trữ."
                    span.set_attribute("memory.exists", False)
                    span.set_attribute("tool.output", output)
                    span.set_status(Status(StatusCode.OK))
                    logger.warning("⚠️ Long-term memory file not found")
                    tool_counter.add(1, {"tool.name": "retrieve_long_term_memory", "status": "no_file"})
                    return output
                
                span.set_attribute("memory.exists", True)
                
                with open(self.longterm_file, 'r', encoding='utf-8') as f:
                    longterm_content = f.read().strip()

                with open(self.longterm_temp_file, 'r', encoding='utf-8') as f:
                    longterm_temp_content = f.read().strip()

                longterm_content = longterm_content + "\n" + longterm_temp_content

                span.set_attribute("memory.content_length", len(longterm_content))
                
                if not longterm_content:
                    output = "Long-term memory trống, chưa có thông tin nào được lưu."
                    span.set_attribute("tool.output", output)
                    span.set_status(Status(StatusCode.OK))
                    logger.info("ℹ️ Long-term memory is empty")
                    tool_counter.add(1, {"tool.name": "retrieve_long_term_memory", "status": "empty"})
                    return output
                
                if self.openai_client is None:
                    output = f"Thông tin từ long-term memory:\n{longterm_content}"
                    span.set_attribute("llm.used", False)
                    span.set_attribute("tool.output", output[:500])  # Limit to 500 chars
                    span.set_status(Status(StatusCode.OK))
                    logger.info("✅ Returning raw long-term memory (no LLM)")
                    tool_counter.add(1, {"tool.name": "retrieve_long_term_memory", "status": "success_no_llm"})
                    return output
            
                prompt_text = f"""
Bạn là một trợ lý thông minh. Nhiệm vụ của bạn là phân tích long-term memory và tìm các thông tin liên quan đến ngữ cảnh hiện tại.

NGỮ CẢNH HIỆN TẠI:
{query}

LONG-TERM MEMORY:
{longterm_content}

Hãy:
1. Tìm các thông tin trong long-term memory có liên quan đến ngữ cảnh hiện tại
2. Tóm tắt lại thành một đoạn ngắn gọn, dễ hiểu
3. Chỉ đưa ra thông tin thực sự hữu ích và liên quan

TÓM TẮT:
"""
                
                span.set_attribute("llm.used", True)
                span.set_attribute("llm.model", self.model_name)
                span.set_attribute("llm.prompt_length", len(prompt_text))
                
                # Nested span for LLM call
                with tracer.start_as_current_span("openai.chat.completions") as llm_span:
                    try:
                        logger.info(f"🤖 Calling OpenAI API (model: {self.model_name})...")
                        
                        # Sử dụng OpenAI client trực tiếp
                        response = self.openai_client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content": prompt_text}
                            ],
                            temperature=0.7
                        )
                        
                        # Add LLM metrics to span
                        llm_span.set_attribute("llm.response.tokens.prompt", response.usage.prompt_tokens)
                        llm_span.set_attribute("llm.response.tokens.completion", response.usage.completion_tokens)
                        llm_span.set_attribute("llm.response.tokens.total", response.usage.total_tokens)
                        llm_span.set_attribute("llm.response.model", response.model)
                        llm_span.set_status(Status(StatusCode.OK))
                        
                        # Trích xuất content từ response
                        content = response.choices[0].message.content
                        llm_span.set_attribute("llm.response.content_length", len(content))
                        
                        span.set_attribute("tool.output_length", len(content))
                        span.set_attribute("tool.output", content[:500])  # Limit to 500 chars
                        span.set_status(Status(StatusCode.OK))
                        logger.info(f"✅ Long-term memory retrieved successfully (tokens: {response.usage.total_tokens})")
                        tool_counter.add(1, {"tool.name": "retrieve_long_term_memory", "status": "success_with_llm"})
                        
                        return content
                        
                    except Exception as llm_error:
                        llm_span.set_status(Status(StatusCode.ERROR, str(llm_error)))
                        llm_span.record_exception(llm_error)
                        logger.error(f"❌ OpenAI API error: {str(llm_error)}")
                        tool_counter.add(1, {"tool.name": "retrieve_long_term_memory", "status": "llm_error"})
                        return f"Thông tin từ long-term memory:\n{longterm_content}"
                
            except Exception as e:
                import traceback
                error_msg = f"Lỗi khi truy vấn long-term memory: {str(e)}"
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                tool_counter.add(1, {"tool.name": "retrieve_long_term_memory", "status": "error"})
                return error_msg
    
    async def _arun(self, query: str) -> str:
        """Async version (không bắt buộc)"""
        return self._run(query)


# ============================================================================
# Tool 2: Save Memory
# ============================================================================

class SaveMemoryInput(BaseModel):
    """Input schema cho Save Memory tool"""
    information: str = Field(
        description="""
Thông tin quan trọng cần lưu vào long-term memory về người dùng.
"""
    )


class SaveMemoryTool(BaseTool):
    """Tool để lưu thông tin quan trọng vào long-term memory (file + Qdrant)"""
    
    name: str = "save_memory"
    description: str = """
    Lưu trữ thông tin dài hạn và quan trọng về người dùng vào long-term memory.
    Chỉ gọi tool này khi xuất hiện thông tin thật sự quan trọng, ổn định theo thời gian và hữu ích cho cá nhân hóa các tương tác sau này.
    
    Thông tin sẽ được lưu vào:
    1. File longterm.txt (với timestamp)
    2. Qdrant database (để semantic search)
    
    Input: Thông tin đã được trích xuất và tóm tắt ngắn gọn
    Output: Xác nhận đã lưu
    
    Ví dụ sử dụng:
    - User: "Tôi tên là Minh, 45 tuổi" → save_memory("Người dùng tên là Minh, 45 tuổi")
    - User: "Tôi bị tiểu đường type 2" → save_memory("Bị bệnh tiểu đường type 2")
    - User: "Con tôi 5 tuổi tên Linh" → save_memory("Có con gái 5 tuổi tên Linh")
    """
    args_schema: Type[BaseModel] = SaveMemoryInput
    longterm_file: str = "longterm_temp.txt"
    llm: Optional[ChatOpenAI] = None
    qdrant_client: Optional[QdrantClient] = None
    protonx_client: Optional[ProtonX] = None
    
    def _run(self, information: str) -> str:
        """Lưu thông tin vào long-term memory"""
        # Get tracer for this tool
        tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)
        
        # Create counter for tool invocations
        tool_counter = meter.create_counter(
            name="tool.invocations",
            description="Number of tool invocations",
            unit="1"
        )
        
        # Start span for this tool execution
        with tracer.start_as_current_span(
            "save_memory",
            attributes={
                "tool.name": "save_memory",
                "tool.input.information": information[:100] if information else "",  # Limit length
                "tool.file": self.longterm_file
            }
        ) as span:
            try:
                logger.info(f"💾 Saving to long-term memory: {information[:50]}...")
                tool_counter.add(1, {"tool.name": "save_memory", "status": "started"})
                
                if not information or not information.strip():
                    output = "Không có thông tin để lưu."
                    span.set_attribute("save.success", False)
                    span.set_attribute("save.reason", "empty_information")
                    span.set_attribute("tool.output", output)
                    span.set_status(Status(StatusCode.OK))
                    logger.warning("⚠️ No information to save (empty)")
                    tool_counter.add(1, {"tool.name": "save_memory", "status": "empty"})
                    return output
                
                span.set_attribute("save.information_length", len(information))
                timestamp = self._get_timestamp()
                span.set_attribute("save.timestamp", timestamp)
                
                # Save to file
                with tracer.start_as_current_span("save_to_file") as file_span:
                    try:
                        with open(self.longterm_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{timestamp}] {information.strip()}\n")
                        file_span.set_attribute("save.file.success", True)
                        file_span.set_status(Status(StatusCode.OK))
                        logger.info(f"✅ Saved to file: {self.longterm_file}")
                    except Exception as e:
                        file_span.set_status(Status(StatusCode.ERROR, str(e)))
                        file_span.record_exception(e)
                        logger.error(f"❌ Error saving to file: {str(e)}")
                        raise
                
                # Save to Qdrant database
                if self.qdrant_client and self.protonx_client:
                    with tracer.start_as_current_span("save_to_qdrant") as qdrant_span:
                        try:
                            logger.info("🔤 Creating embedding for Qdrant...")
                            
                            # Create embedding
                            response = self.protonx_client.embeddings.create([information.strip()])
                            if isinstance(response, dict):
                                embedding = response["data"][0]["embedding"]
                            else:
                                embedding = response.data[0].embedding
                            
                            qdrant_span.set_attribute("embedding.dimension", len(embedding))
                            
                            # Get next ID (count existing points + 1)
                            collection_info = self.qdrant_client.get_collection(LONGTERM_COLLECTION_NAME)
                            next_id = collection_info.points_count + 1
                            
                            # Create point
                            point = PointStruct(
                                id=next_id,
                                vector={"default": embedding},
                                payload={
                                    "text": f"[{timestamp}] {information.strip()}",
                                    "text_without_timestamp": information.strip(),
                                    "timestamp": timestamp,
                                    "created_at": datetime.now().isoformat()
                                }
                            )
                            
                            # Upload to Qdrant
                            self.qdrant_client.upsert(
                                collection_name=LONGTERM_COLLECTION_NAME,
                                points=[point]
                            )
                            
                            qdrant_span.set_attribute("save.qdrant.success", True)
                            qdrant_span.set_attribute("save.qdrant.point_id", next_id)
                            qdrant_span.set_status(Status(StatusCode.OK))
                            logger.info(f"✅ Saved to Qdrant database (ID: {next_id})")
                            
                        except Exception as e:
                            qdrant_span.set_status(Status(StatusCode.ERROR, str(e)))
                            qdrant_span.record_exception(e)
                            logger.warning(f"⚠️ Could not save to Qdrant: {str(e)}")
                            # Don't fail if Qdrant save fails - file save is primary
                else:
                    logger.warning("⚠️ Qdrant/ProtonX not available, skipping database save")
                
                output = f"Đã lưu: {information.strip()}"
                span.set_attribute("save.success", True)
                span.set_attribute("tool.output", output)
                span.set_status(Status(StatusCode.OK))
                logger.info(f"✅ Successfully saved to long-term memory")
                tool_counter.add(1, {"tool.name": "save_memory", "status": "success"})
                
                return output
                
            except Exception as e:
                span.set_attribute("save.success", False)
                span.set_attribute("save.error", str(e))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"❌ Error saving to long-term memory: {str(e)}")
                tool_counter.add(1, {"tool.name": "save_memory", "status": "error"})
                return f"Lỗi khi lưu: {str(e)}"
    
    def _get_timestamp(self):
        """Lấy timestamp hiện tại"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    async def _arun(self, information: str) -> str:
        """Async version"""
        return self._run(information)

