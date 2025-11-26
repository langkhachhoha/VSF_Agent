import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from agent_with_memory import MemoryAgent

# Setup logger
logger = logging.getLogger(__name__)

# Import telemetry configuration (must be before other imports that use OpenTelemetry)
try:
    from telemetry_config import setup_telemetry
    from opentelemetry import trace, metrics
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.trace import Status, StatusCode
    
    tracer = trace.get_tracer("api_server")
    meter = metrics.get_meter("api_server")
    TELEMETRY_ENABLED = True
    logger.info("✅ OpenTelemetry enabled for API server")
except ImportError as e:
    TELEMETRY_ENABLED = False
    tracer = None
    meter = None
    logger.warning(f"⚠️ OpenTelemetry not available: {e}")

# ============================================================================
# Configuration
# ============================================================================

from config import OPENAI_API_KEY
LONGTERM_FILE = "longterm.txt"

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Memory Agent API",
    description="API cho Memory Agent với long-term memory và doctor retrieval",
    version="1.0.0"
)

# Instrument FastAPI with OpenTelemetry
if TELEMETRY_ENABLED:
    FastAPIInstrumentor.instrument_app(app)
    logger.info("✅ FastAPI instrumented with OpenTelemetry")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global Agent Instance & Tool Call History
# ============================================================================

agent: Optional[MemoryAgent] = None
tool_call_history: List[Dict] = []

def get_agent() -> MemoryAgent:
    """Lấy hoặc khởi tạo agent instance"""
    global agent
    if agent is None:
        agent = MemoryAgent(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4o-mini",
            buffer_size=10,
            longterm_file=LONGTERM_FILE
        )
    return agent

# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ToolCall(BaseModel):
    tool_name: str
    tool_input: Dict
    tool_output: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: Optional[List[ToolCall]] = []

class MemoryResponse(BaseModel):
    content: str

class BufferMemoryResponse(BaseModel):
    messages: List[Dict[str, str]]

class StatusResponse(BaseModel):
    status: str
    message: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Memory Agent API",
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat với agent (có auto-priming)
    
    Args:
        request: ChatRequest với message và session_id
        
    Returns:
        ChatResponse với response từ agent và danh sách tools đã sử dụng
    """
    try:
        global tool_call_history
        agent_instance = get_agent()
        
        # Sử dụng agent.chat() thay vì agent_executor.invoke() để có priming
        with tracer.start_as_current_span("chat_with_agent") as chat_span:
            response_text = agent_instance.chat(request.message, auto_prime=True)
            chat_span.set_attribute("chat_with_agent.value", response_text)
        
        # Note: Khi dùng agent.chat(), không có intermediate_steps
        # Nếu cần track tools, phải modify agent.chat() để return cả intermediate_steps
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            tools_used=[]  # Tạm thời empty, có thể enhance sau
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi chat: {str(e)}")

@app.get("/memory/longterm", response_model=MemoryResponse)
async def get_longterm_memory():
    """
    Lấy nội dung long-term memory
    
    Returns:
        MemoryResponse với nội dung long-term memory
    """
    try:
        agent_instance = get_agent()
        content = agent_instance.view_longterm_memory()
        
        return MemoryResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy long-term memory: {str(e)}")

@app.get("/memory/buffer", response_model=BufferMemoryResponse)
async def get_buffer_memory():
    """
    Lấy nội dung buffer memory
    
    Returns:
        BufferMemoryResponse với danh sách messages
    """
    try:
        agent_instance = get_agent()
        messages = agent_instance.view_buffer_memory()
        
        formatted_messages = []
        for msg in messages:
            role = "user" if msg.type == "human" else "assistant"
            formatted_messages.append({
                "role": role,
                "content": msg.content
            })
        
        return BufferMemoryResponse(messages=formatted_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy buffer memory: {str(e)}")

@app.delete("/memory/longterm", response_model=StatusResponse)
async def clear_longterm_memory():
    """
    Xóa long-term memory
    
    Returns:
        StatusResponse với trạng thái
    """
    try:
        agent_instance = get_agent()
        agent_instance.clear_longterm_memory()
        
        return StatusResponse(
            status="success",
            message="Đã xóa long-term memory"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa long-term memory: {str(e)}")

@app.delete("/memory/buffer", response_model=StatusResponse)
async def clear_buffer_memory():
    """
    Xóa buffer memory
    
    Returns:
        StatusResponse với trạng thái
    """
    try:
        agent_instance = get_agent()
        agent_instance.clear_buffer_memory()
        
        return StatusResponse(
            status="success",
            message="Đã xóa buffer memory"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa buffer memory: {str(e)}")

@app.get("/tools/history")
async def get_tool_history():
    """
    Lấy lịch sử tool calls
    
    Returns:
        Danh sách các tool calls đã thực hiện
    """
    global tool_call_history
    return {
        "total_calls": len(tool_call_history),
        "history": tool_call_history
    }

@app.delete("/tools/history")
async def clear_tool_history():
    """
    Xóa lịch sử tool calls
    
    Returns:
        StatusResponse
    """
    global tool_call_history
    tool_call_history = []
    return StatusResponse(
        status="success",
        message="Đã xóa lịch sử tool calls"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        agent_instance = get_agent()
        return {
            "status": "healthy",
            "agent_initialized": agent_instance is not None,
            "longterm_file": LONGTERM_FILE,
            "longterm_exists": os.path.exists(LONGTERM_FILE),
            "total_tool_calls": len(tool_call_history),
            "is_primed": agent_instance.is_primed if agent_instance else False,
            "messages_since_prime": agent_instance.message_count_since_prime if agent_instance else 0,
            "buffer_size": agent_instance.buffer_size if agent_instance else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/priming/status")
async def get_priming_status():
    """
    Lấy trạng thái priming của agent
    
    Returns:
        Thông tin về priming status
    """
    try:
        agent_instance = get_agent()
        return {
            "is_primed": agent_instance.is_primed,
            "message_count_since_prime": agent_instance.message_count_since_prime,
            "buffer_size": agent_instance.buffer_size,
            "should_reprime": agent_instance._should_reprime(),
            "messages_until_reprime": max(0, agent_instance.buffer_size - agent_instance.message_count_since_prime)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy priming status: {str(e)}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Memory Agent API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

