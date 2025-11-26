"""
Doctor Tool: Retrieve doctors from Qdrant database
"""

import logging
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from protonx import ProtonX

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from config import COLLECTION_NAME

logger = logging.getLogger(__name__)


# ============================================================================
# Tool: Retrieve Doctor
# ============================================================================

class RetrieveDoctorInput(BaseModel):
    """Input schema cho Retrieve Doctor tool"""
    query: str = Field(
        description="Mô tả tình trạng bệnh, triệu chứng hoặc chuyên khoa cần tìm (ví dụ: 'bác sĩ tim mạch', 'đau đầu thường xuyên', 'bệnh tiểu đường')"
    )
    top_k: int = Field(
        default=3,
        description="Số lượng bác sĩ cần trả về (mặc định: 3)"
    )


class RetrieveDoctorTool(BaseTool):
    """Tool để tìm kiếm bác sĩ phù hợp từ database Qdrant"""
    
    name: str = "retrieve_doctor"
    description: str = """
    Sử dụng tool này để tìm kiếm bác sĩ phù hợp với tình trạng bệnh hoặc chuyên khoa.
    Tool sẽ tìm kiếm trong database và trả về thông tin bác sĩ bao gồm: tên, chuyên môn, nơi làm việc, giới thiệu.
    Input: Mô tả bệnh tình hoặc chuyên khoa cần tìm
    Output: Danh sách bác sĩ phù hợp nhất
    """
    args_schema: Type[BaseModel] = RetrieveDoctorInput
    qdrant_client: Optional[QdrantClient] = None
    protonx_client: Optional[ProtonX] = None
    
    def _run(self, query: str, top_k: int = 3) -> str:
        """Thực thi tool để tìm kiếm bác sĩ"""
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
            "retrieve_doctor",
            attributes={
                "tool.name": "retrieve_doctor",
                "tool.input.query": query[:500] if query else "",  # Limit to 500 chars
                "tool.input.top_k": top_k,
                "db.collection": COLLECTION_NAME
            }
        ) as span:
            try:
                logger.info(f"🔍 Searching for doctors with query: {query[:50]}... (top_k={top_k})")
                tool_counter.add(1, {"tool.name": "retrieve_doctor", "status": "started"})
                
                if self.qdrant_client is None or self.protonx_client is None:
                    output = "Lỗi: Không thể kết nối đến database."
                    span.set_attribute("db.connected", False)
                    span.set_attribute("tool.output", output)
                    span.set_status(Status(StatusCode.ERROR, "Database not connected"))
                    logger.error("❌ Database connection not available")
                    tool_counter.add(1, {"tool.name": "retrieve_doctor", "status": "no_connection"})
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
                        tool_counter.add(1, {"tool.name": "retrieve_doctor", "status": "embedding_error"})
                        return f"Lỗi khi tạo embedding: {str(e)}"
                
                # Search in Qdrant
                with tracer.start_as_current_span("qdrant_search") as search_span:
                    try:
                        logger.info(f"🔎 Searching in Qdrant collection: {COLLECTION_NAME}...")
                        search_span.set_attribute("db.operation", "search")
                        search_span.set_attribute("db.collection", COLLECTION_NAME)
                        search_span.set_attribute("db.limit", top_k)
                        
                        hits = self.qdrant_client.search(
                            collection_name=COLLECTION_NAME,
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
                        tool_counter.add(1, {"tool.name": "retrieve_doctor", "status": "search_error"})
                        return f"Lỗi khi tìm kiếm: {str(e)}"
                
                if not hits:
                    output = "Không tìm thấy bác sĩ phù hợp."
                    span.set_attribute("results.found", False)
                    span.set_attribute("tool.output", output)
                    span.set_status(Status(StatusCode.OK))
                    logger.warning("⚠️ No doctors found")
                    tool_counter.add(1, {"tool.name": "retrieve_doctor", "status": "no_results"})
                    return output
                
                span.set_attribute("results.found", True)
                span.set_attribute("results.count", len(hits))
                
                # Format results
                results = []
                for i, hit in enumerate(hits, 1):
                    ten_bac_si = hit.payload.get('ten_bac_si', 'N/A')
                    chuyen_mon = hit.payload.get('chuyen_mon', [])
                    noi_lam_viec = hit.payload.get('noi_lam_viec', 'N/A')
                    gioi_thieu = hit.payload.get('gioi_thieu', 'N/A')
                    
                    if len(gioi_thieu) > 300:
                        gioi_thieu = gioi_thieu[:300] + "..."
                    
                    chuyen_mon_str = ", ".join(chuyen_mon) if chuyen_mon else "Không có thông tin"
                    
                    result = f"""
{i}. Bác sĩ: {ten_bac_si}
   Chuyên môn: {chuyen_mon_str}
   Nơi làm việc: {noi_lam_viec}
   Giới thiệu: {gioi_thieu}
"""
                    results.append(result)
                
                final_result = "\n".join(results)
                span.set_attribute("results.output_length", len(final_result))
                span.set_attribute("tool.output", final_result[:500])  # Limit to 500 chars for attribute
                span.set_status(Status(StatusCode.OK))
                logger.info(f"✅ Successfully retrieved {len(hits)} doctors")
                tool_counter.add(1, {"tool.name": "retrieve_doctor", "status": "success"})
                
                return final_result
                
            except Exception as e:
                import traceback
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"❌ Error in retrieve_doctor: {str(e)}\n{traceback.format_exc()}")
                tool_counter.add(1, {"tool.name": "retrieve_doctor", "status": "error"})
                return f"Lỗi khi tìm kiếm bác sĩ: {str(e)}"
    
    async def _arun(self, query: str, top_k: int = 3) -> str:
        """Async version"""
        return self._run(query, top_k)

