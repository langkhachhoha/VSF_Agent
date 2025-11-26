import json
from qdrant_client import QdrantClient
from protonx import ProtonX

# ===================== CONFIG =====================
QDRANT_URL = "https://6b21144b-609e-4f90-a884-7b27d70f2d97.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.X_8203L06c_QSb1qPXh7Z0Z6jxk5ZplN6oJVlrYaLug"
COLLECTION_NAME = "doctor_vinmec"

# Khởi tạo clients
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
protonx_client = ProtonX(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImhhNzQ2NjUyNEBnbWFpbC5jb20iLCJpYXQiOjE3NjMwMzA5NDEsImV4cCI6MTc2NTYyMjk0MX0.OYigPmHvFoTdPCPulH1101pFeJxuJsElTPHR_cNotSg")

def get_embedding(text):
    """Tạo embedding cho text"""
    try:
        response = protonx_client.embeddings.create([text])
        if isinstance(response, dict):
            return response["data"][0]["embedding"]
        else:
            return response.data[0].embedding
    except Exception as e:
        print(f"❌ Lỗi khi tạo embedding: {str(e)[:200]}")
        return None

def search_doctors(query, top_k=5):
    """
    Tìm kiếm bác sĩ theo query
    
    Args:
        query: Câu hỏi/yêu cầu tìm kiếm
        top_k: Số lượng kết quả trả về
    
    Returns:
        List các kết quả tìm kiếm
    """
    print(f"\n🔍 Tìm kiếm: '{query}'")
    print("=" * 80)
    
    # Tạo embedding cho query
    query_emb = get_embedding(query)
    if query_emb is None:
        return []
    
    # Tìm kiếm trong Qdrant
    try:
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=("default", query_emb),
            limit=top_k
        )
        
        # Hiển thị kết quả
        results = []
        for i, hit in enumerate(hits, 1):
            result = {
                "rank": i,
                "score": hit.score,
                "ten_bac_si": hit.payload.get('ten_bac_si', 'N/A'),
                "chuyen_mon": hit.payload.get('chuyen_mon', []),
                "noi_lam_viec": hit.payload.get('noi_lam_viec', 'N/A'),
                "url": hit.payload.get('url', 'N/A')
            }
            results.append(result)
            
            # In kết quả
            print(f"\n{i}. 🏆 Score: {hit.score:.3f}")
            print(f"   👨‍⚕️ Tên: {result['ten_bac_si']}")
            
            # Hiển thị chuyên môn
            if result['chuyen_mon']:
                chuyen_mon_str = ", ".join(result['chuyen_mon'])
                print(f"   💼 Chuyên môn: {chuyen_mon_str}")
            else:
                print(f"   💼 Chuyên môn: Không có thông tin")
            
            print(f"   🏥 Nơi làm việc: {result['noi_lam_viec']}")
            print(f"   🔗 URL: {result['url']}")
            print("-" * 80)
        
        return results
        
    except Exception as e:
        print(f"❌ Lỗi khi tìm kiếm: {str(e)[:200]}")
        return []

def run_test_queries():
    """Chạy một loạt các query test"""
    
    print("=" * 80)
    print("🧪 TEST CHỨC NĂNG TÌM KIẾM BÁC SĨ")
    print("=" * 80)
    
    # Danh sách các query test
    test_queries = [
        {
            "query": "Bác sĩ chuyên khoa tim mạch",
            "description": "Tìm bác sĩ tim mạch"
        },
        {
            "query": "Bác sĩ sản phụ khoa giỏi",
            "description": "Tìm bác sĩ sản phụ khoa"
        },
        {
            "query": "Bác sĩ nhi khoa có kinh nghiệm",
            "description": "Tìm bác sĩ nhi khoa"
        },
        {
            "query": "Bác sĩ chuyên về tiêu hóa",
            "description": "Tìm bác sĩ tiêu hóa"
        },
        {
            "query": "Bác sĩ chuyên điều trị ung thư",
            "description": "Tìm bác sĩ ung thư"
        },
        {
            "query": "Bác sĩ phẫu thuật thẩm mỹ",
            "description": "Tìm bác sĩ thẩm mỹ"
        },
        {
            "query": "Bác sĩ chuyên về xương khớp",
            "description": "Tìm bác sĩ xương khớp"
        },
        {
            "query": "Bác sĩ da liễu giỏi ở Hà Nội",
            "description": "Tìm bác sĩ da liễu"
        },
        {
            "query": "Bác sĩ gây mê hồi sức",
            "description": "Tìm bác sĩ gây mê"
        },
        {
            "query": "Bác sĩ chuyên khoa mắt",
            "description": "Tìm bác sĩ nhãn khoa"
        }
    ]
    
    # Lưu kết quả
    all_results = {}
    
    # Chạy từng query
    for idx, test in enumerate(test_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST {idx}/{len(test_queries)}: {test['description']}")
        print(f"{'='*80}")
        
        results = search_doctors(test['query'], top_k=3)
        all_results[test['query']] = results
        
        # Chờ một chút để tránh rate limit
        if idx < len(test_queries):
            import time
            print("\n⏳ Chờ 2s trước query tiếp theo...")
            time.sleep(2)
    
    # Lưu kết quả vào file JSON
    output_file = "/Users/apple/VITA /Doctor_vinmec/Database/test_search_results.json"
    print(f"\n\n{'='*80}")
    print(f"💾 Lưu kết quả test vào {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã lưu kết quả test!")
    print(f"{'='*80}")
    
    # Tóm tắt
    print(f"\n\n📊 TÓM TẮT KẾT QUẢ TEST:")
    print(f"{'='*80}")
    print(f"✅ Tổng số query test: {len(test_queries)}")
    print(f"✅ Tổng số kết quả: {sum(len(results) for results in all_results.values())}")
    print(f"{'='*80}")

def interactive_search():
    """Chế độ tìm kiếm tương tác"""
    
    print("\n" + "="*80)
    print("🔍 CHẾ ĐỘ TÌM KIẾM TƯƠNG TÁC")
    print("="*80)
    print("Nhập câu hỏi để tìm kiếm bác sĩ (hoặc 'exit' để thoát)")
    print("="*80)
    
    while True:
        try:
            query = input("\n💬 Nhập câu hỏi: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Tạm biệt!")
                break
            
            if not query:
                print("⚠️ Vui lòng nhập câu hỏi!")
                continue
            
            # Tìm kiếm
            search_doctors(query, top_k=5)
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("🏥 TEST SEARCH - HỆ THỐNG TÌM KIẾM BÁC SĨ VINMEC")
    print("="*80)
    
    # Kiểm tra collection
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME in collections:
            print(f"✅ Đã kết nối với collection '{COLLECTION_NAME}'")
            
            # Lấy thông tin collection
            collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
            print(f"📊 Số lượng bác sĩ trong database: {collection_info.points_count}")
        else:
            print(f"❌ Không tìm thấy collection '{COLLECTION_NAME}'")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Lỗi khi kết nối Qdrant: {str(e)}")
        sys.exit(1)
    
    # Menu
    print("\n" + "="*80)
    print("CHỌN CHẾ ĐỘ:")
    print("1. Chạy test tự động (10 queries mẫu)")
    print("2. Tìm kiếm tương tác")
    print("="*80)
    
    choice = input("\nNhập lựa chọn (1 hoặc 2): ").strip()
    
    if choice == "1":
        run_test_queries()
    elif choice == "2":
        interactive_search()
    else:
        print("❌ Lựa chọn không hợp lệ!")

