"""
Long-term Memory Qdrant Tool: Semantic search in long-term memory database
"""

import logging
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from protonx import ProtonX

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from config import LONGTERM_COLLECTION_NAME

logger = logging.getLogger(__name__)


# ============================================================================
# Tool: Retrieve Qdrant LongTerm
# ============================================================================

class RetrieveQdrantLongTermInput(BaseModel):
    """Input schema cho Retrieve Qdrant LongTerm tool"""
    query: str = Field(
        description="Câu hỏi hoặc ngữ cảnh để tìm kiếm thông tin liên quan trong long-term memory database"
    )
    top_k: int = Field(
        default=3,
        description="Số lượng kết quả cần trả về (mặc định: 3)"
    )


class RetrieveQdrantLongTermTool(BaseTool):
    """Tool để tìm kiếm semantic trong long-term memory database"""
    
    name: str = "retrieve_qdrant_longterm"
    description: str = """
    Sử dụng tool này khi thông tin long-term memory hiện có KHÔNG ĐỦ để trả lời câu hỏi.
    Tool sẽ thực hiện semantic search trong database long-term memory để tìm thông tin liên quan.
    
    Khi nào sử dụng:
    - Khi cần tìm thông tin cụ thể về người dùng mà chưa có trong context hiện tại
    - Khi người dùng hỏi về thông tin đã lưu trước đó nhưng không có trong buffer memory
    - Khi cần tra cứu lịch sử thông tin chi tiết
    
    Input: Câu hỏi hoặc mô tả thông tin cần tìm
    Output: Top K thông tin liên quan nhất từ long-term memory
    """
    args_schema: Type[BaseModel] = RetrieveQdrantLongTermInput
    qdrant_client: Optional[QdrantClient] = None
    protonx_client: Optional[ProtonX] = None
    
    def _run(self, query: str, top_k: int = 3) -> str:
        """Thực thi tool để tìm kiếm trong long-term memory database"""
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
            "retrieve_qdrant_longterm",
            attributes={
                "tool.name": "retrieve_qdrant_longterm",
                "tool.input.query": query[:500] if query else "",  # Limit to 500 chars
                "tool.input.top_k": top_k,
                "db.collection": LONGTERM_COLLECTION_NAME
            }
        ) as span:
            try:
                logger.info(f"🔍 Searching long-term memory with query: {query[:50]}... (top_k={top_k})")
                tool_counter.add(1, {"tool.name": "retrieve_qdrant_longterm", "status": "started"})
                
                if self.qdrant_client is None or self.protonx_client is None:
                    output = "Lỗi: Không thể kết nối đến long-term memory database."
                    span.set_attribute("db.connected", False)
                    span.set_attribute("tool.output", output)
                    span.set_status(Status(StatusCode.ERROR, "Database not connected"))
                    logger.error("❌ Database connection not available")
                    tool_counter.add(1, {"tool.name": "retrieve_qdrant_longterm", "status": "no_connection"})
                    return output
                
                span.set_attribute("db.connected", True)
                
                # Create embedding
                with tracer.start_as_current_span("create_embedding") as emb_span:
                    try:
                        logger.info("🔤 Creating query embedding...")
                        response = self.protonx_client.embeddings.create([query])
                        if isinstance(response, dict):
                            query_emb = response["data"][0]["embedding"]
                        else:
                            query_emb = response.data[0].embedding
                        
                        emb_span.set_attribute("embedding.dimension", len(query_emb))
                        emb_span.set_status(Status(StatusCode.OK))
                        logger.info(f"✅ Embedding created (dimension: {len(query_emb)})")
                        
                    except Exception as e:
                        emb_span.set_status(Status(StatusCode.ERROR, str(e)))
                        emb_span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        logger.error(f"❌ Error creating embedding: {str(e)}")
                        tool_counter.add(1, {"tool.name": "retrieve_qdrant_longterm", "status": "embedding_error"})
                        return f"Lỗi khi tạo embedding: {str(e)}"
                
                # Search in Qdrant
                with tracer.start_as_current_span("qdrant_search") as search_span:
                    try:
                        logger.info(f"🔎 Searching in Qdrant collection: {LONGTERM_COLLECTION_NAME}...")
                        search_span.set_attribute("db.operation", "search")
                        search_span.set_attribute("db.collection", LONGTERM_COLLECTION_NAME)
                        search_span.set_attribute("db.limit", top_k)
                        
                        hits = self.qdrant_client.search(
                            collection_name=LONGTERM_COLLECTION_NAME,
                            query_vector=("default", query_emb),
                            limit=top_k
                        )
                        
                        search_span.set_attribute("db.results.count", len(hits))
                        search_span.set_status(Status(StatusCode.OK))
                        logger.info(f"✅ Found {len(hits)} results")
                        
                    except Exception as e:
                        search_span.set_status(Status(StatusCode.ERROR, str(e)))
                        search_span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        logger.error(f"❌ Error searching Qdrant: {str(e)}")
                        tool_counter.add(1, {"tool.name": "retrieve_qdrant_longterm", "status": "search_error"})
                        return f"Lỗi khi tìm kiếm: {str(e)}"
                
                if not hits:
                    output = "Không tìm thấy thông tin liên quan trong long-term memory."
                    span.set_attribute("results.found", False)
                    span.set_attribute("tool.output", output)
                    span.set_status(Status(StatusCode.OK))
                    logger.warning("⚠️ No long-term memory found")
                    tool_counter.add(1, {"tool.name": "retrieve_qdrant_longterm", "status": "no_results"})
                    return output
                
                span.set_attribute("results.found", True)
                span.set_attribute("results.count", len(hits))
                
                # Format results
                results = []
                for i, hit in enumerate(hits, 1):
                    text = hit.payload.get('text_without_timestamp', 'N/A')
                    timestamp = hit.payload.get('timestamp', 'unknown')
                    score = hit.score
                    
                    result = f"{i}. [{timestamp}] {text} (relevance: {score:.3f})"
                    results.append(result)
                    
                    # Log individual result
                    logger.info(f"  Result {i}: score={score:.3f}, text={text[:50]}...")
                
                final_result = "\n".join(results)
                output = f"Thông tin từ long-term memory database:\n{final_result}"
                
                span.set_attribute("results.output_length", len(final_result))
                span.set_attribute("tool.output", output[:500])  # Limit to 500 chars for attribute
                span.set_status(Status(StatusCode.OK))
                logger.info(f"✅ Successfully retrieved {len(hits)} memory entries")
                tool_counter.add(1, {"tool.name": "retrieve_qdrant_longterm", "status": "success"})
                
                return output
                
            except Exception as e:
                import traceback
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"❌ Error in retrieve_qdrant_longterm: {str(e)}\n{traceback.format_exc()}")
                tool_counter.add(1, {"tool.name": "retrieve_qdrant_longterm", "status": "error"})
                return f"Lỗi khi tìm kiếm long-term memory: {str(e)}"
    
    async def _arun(self, query: str, top_k: int = 3) -> str:
        """Async version"""
        return self._run(query, top_k)

