import uuid
import json
import time
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from protonx import ProtonX 
import os

from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, PROTONX_API_KEY
# ===================== CONFIG =====================
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ===================== STEP 1. Đọc file JSON chứa thông tin bác sĩ =====================
def read_doctors_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        doctors = json.load(f)
    print(f"📖 Đã đọc {len(doctors)} bác sĩ từ file JSON")
    return doctors

# ===================== STEP 2. Format thông tin bác sĩ =====================
def format_doctor_info(doctor):
    """Format thông tin bác sĩ theo dạng: Chuyên môn, Nơi làm việc, Giới thiệu"""
    chuyen_mon = ", ".join(doctor.get("chuyen_mon", [])) if doctor.get("chuyen_mon") else "Không có thông tin"
    noi_lam_viec = doctor.get("noi_lam_viec", "Không có thông tin")
    gioi_thieu = doctor.get("gioi_thieu", "Không có thông tin")
    
    formatted_text = f"""Chuyên môn: {chuyen_mon}

Nơi làm việc: {noi_lam_viec}

Giới thiệu: {gioi_thieu}"""
    
    return formatted_text

# ===================== STEP 3. Tạo batch embedding =====================
protonx_client = ProtonX(api_key=PROTONX_API_KEY)

def get_batch_embeddings(texts, max_retries=5):
    """
    Tạo embedding cho nhiều text cùng lúc (batch)
    
    Args:
        texts: List các text cần tạo embedding
        max_retries: Số lần thử lại
    
    Returns:
        List các embedding vectors
    """
    for attempt in range(max_retries):
        try:
            # Gọi API với batch texts
            response = protonx_client.embeddings.create(texts)
            
            # ProtonX trả về dict với nhiều embeddings
            if isinstance(response, dict):
                embeddings = [item["embedding"] for item in response["data"]]
                return embeddings
            else:
                embeddings = [item.embedding for item in response.data]
                return embeddings
                
        except Exception as e:
            error_str = str(e)
            # Kiểm tra nếu là lỗi rate limit (429 hoặc TOKEN_LIMIT_EXCEEDED)
            if "429" in error_str or "rate limit" in error_str.lower() or "per-minute" in error_str.lower():
                if attempt < max_retries - 1:
                    print(f"⚠️ Đạt rate limit! Chờ 60s...")
                    time.sleep(60)
                else:
                    print(f"❌ Vẫn bị rate limit sau {max_retries} lần thử")
                    raise
            else:
                # Lỗi khác, thử lại ngay
                if attempt < max_retries - 1:
                    print(f"⚠️ Lỗi (attempt {attempt + 1}/{max_retries}): {error_str[:150]}")
                    print(f"⏳ Thử lại ngay...")
                else:
                    print(f"❌ Không thể tạo batch embedding sau {max_retries} lần thử")
                    raise
    return None

# ===================== STEP 4. Xóa và tạo lại collection Qdrant =====================
def recreate_collection():
    """Xóa collection cũ (nếu có) và tạo collection mới"""
    collections = [c.name for c in qdrant.get_collections().collections]
    
    # Xóa collection cũ nếu tồn tại
    if COLLECTION_NAME in collections:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🗑️ Đã xóa collection cũ '{COLLECTION_NAME}'")
    
    # Tạo collection mới với named vector
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "default": VectorParams(size=768, distance=Distance.COSINE)
        }
    )
    print(f"✅ Đã tạo collection mới '{COLLECTION_NAME}'")

# ===================== STEP 5. Tạo embeddings và lưu vào file local =====================
def create_and_save_embeddings(doctors, batch_size=100, output_file="doctor_embeddings.json"):
    """
    Tạo embeddings cho tất cả bác sĩ theo batch và lưu vào file local
    
    Args:
        doctors: Danh sách bác sĩ
        batch_size: Số lượng bác sĩ mỗi batch (mỗi request API)
        output_file: Tên file để lưu embeddings
    
    Returns:
        List các doctor data với embeddings
    """
    total = len(doctors)
    all_doctor_data = []
    request_count = 0
    start_time = time.time()
    
    print(f"📊 Tổng số bác sĩ: {total}")
    print(f"📦 Batch size: {batch_size} bác sĩ/request")
    print(f"⚡ Gọi API liên tục (chỉ chờ 60s khi gặp rate limit)")
    print("-" * 60)
    
    # Xử lý theo batch
    for batch_idx in range(0, total, batch_size):
        batch_end = min(batch_idx + batch_size, total)
        batch_doctors = doctors[batch_idx:batch_end]
        batch_num = (batch_idx // batch_size) + 1
        
        print(f"\n🔄 Đang xử lý batch {batch_num}: Bác sĩ {batch_idx + 1} đến {batch_end}...")
        
        try:
            # Format text cho tất cả bác sĩ trong batch
            batch_texts = [format_doctor_info(doctor) for doctor in batch_doctors]
            
            # Tạo embeddings cho cả batch (1 request API duy nhất)
            print(f"📡 Gọi API để tạo {len(batch_texts)} embeddings...")
            batch_embeddings = get_batch_embeddings(batch_texts)
            request_count += 1
            
            if batch_embeddings is None:
                print(f"❌ Không thể tạo embeddings cho batch {batch_num}, bỏ qua...")
                continue
            
            # Kết hợp data với embeddings
            for idx, (doctor, text, embedding) in enumerate(zip(batch_doctors, batch_texts, batch_embeddings)):
                doctor_data = {
                    "doctor_id": batch_idx + idx,
                    "ten_bac_si": doctor.get("ten_bac_si", ""),
                    "chuyen_mon": doctor.get("chuyen_mon", []),
                    "noi_lam_viec": doctor.get("noi_lam_viec", ""),
                    "gioi_thieu": doctor.get("gioi_thieu", ""),
                    "url": doctor.get("url", ""),
                    "text": text,
                    "embedding": embedding
                }
                all_doctor_data.append(doctor_data)
            
            print(f"✅ Đã xử lý batch {batch_num} - Tổng: {batch_end}/{total} bác sĩ ({(batch_end / total * 100):.1f}%)")
                
        except Exception as e:
            error_msg = str(e)[:200]
            print(f"\n❌ Lỗi khi xử lý batch {batch_num}: {error_msg}")
            print("⏩ Bỏ qua batch này và tiếp tục...\n")
            continue
    
    # Lưu vào file JSON
    print(f"\n💾 Đang lưu {len(all_doctor_data)} bác sĩ vào file '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_doctor_data, f, ensure_ascii=False, indent=2)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Đã lưu embeddings vào file!")
    print(f"⏱️  Tổng thời gian: {elapsed_time/60:.1f} phút")
    print(f"📊 Tổng số request API: {request_count}")
    
    return all_doctor_data

# ===================== STEP 6. Upload embeddings từ file lên Qdrant =====================
def upload_embeddings_to_qdrant(doctor_data_list, batch_size=100):
    """
    Upload embeddings từ file local lên Qdrant
    
    Args:
        doctor_data_list: List các doctor data với embeddings
        batch_size: Số lượng points mỗi lần upload lên Qdrant
    """
    total = len(doctor_data_list)
    print(f"\n📤 Bắt đầu upload {total} bác sĩ lên Qdrant...")
    print(f"📦 Upload batch size: {batch_size} points/batch")
    print("-" * 60)
    
    # Upload theo batch
    for batch_idx in range(0, total, batch_size):
        batch_end = min(batch_idx + batch_size, total)
        batch_data = doctor_data_list[batch_idx:batch_end]
        batch_num = (batch_idx // batch_size) + 1
        
        # Tạo points cho Qdrant
        points = []
        for data in batch_data:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={"default": data["embedding"]},
                payload={
                    "text": data["text"],
                    "ten_bac_si": data["ten_bac_si"],
                    "chuyen_mon": data["chuyen_mon"],
                    "noi_lam_viec": data["noi_lam_viec"],
                    "gioi_thieu": data["gioi_thieu"],
                    "url": data["url"],
                    "doctor_id": data["doctor_id"]
                }
            )
            points.append(point)
        
        # Upload batch
        print(f"📤 Uploading batch {batch_num} ({len(points)} bác sĩ)...")
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Đã upload batch {batch_num} - Tổng: {batch_end}/{total} bác sĩ ({(batch_end / total * 100):.1f}%)")
    
    print(f"\n✅ Hoàn thành upload lên Qdrant!")

# ===================== MAIN =====================
if __name__ == "__main__":
    # Đường dẫn file JSON chứa thông tin bác sĩ

    json_file_path = "/Users/apple/VITA /Doctor_vinmec/crawl_data/vinmec_doctors_unique.json"
    embeddings_file = "/Users/apple/VITA /Doctor_vinmec/Database/doctor_embeddings.json"
    
    print("=" * 60)
    print("🏥 BẮT ĐẦU UPLOAD THÔNG TIN BÁC SĨ LÊN QDRANT")
    print("=" * 60)
    
    # Bước 1: Đọc file JSON
    print("\n📖 Bước 1: Đọc file JSON...")
    doctors = read_doctors_json(json_file_path)
    
    # Bước 2: Tạo embeddings và lưu vào file local
    print("\n🤖 Bước 2: Tạo embeddings cho tất cả bác sĩ (batch processing)...")
    doctor_data_list = create_and_save_embeddings(
        doctors, 
        batch_size=5,  # 5 bác sĩ mỗi request API (tránh vượt quá 4096 tokens)
        output_file=embeddings_file
    )
    
    # Bước 3: Xóa và tạo lại collection Qdrant
    print("\n🗑️ Bước 3: Xóa collection cũ và tạo mới...")
    recreate_collection()
    
    # Bước 4: Upload embeddings lên Qdrant
    print("\n🚀 Bước 4: Upload embeddings lên Qdrant...")
    upload_embeddings_to_qdrant(
        doctor_data_list,
        batch_size=100  # 100 points mỗi lần upload
    )
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH! Đã upload thông tin bác sĩ lên Qdrant Cloud.")
    print("=" * 60)
    print(f"💾 File embeddings đã lưu tại: {embeddings_file}")
    
    # Test tìm kiếm
    print("\n🔍 Test tìm kiếm...")
    query = "Bác sĩ chuyên khoa tim mạch"
    print(f"📡 Tạo embedding cho query...")
    query_emb = get_batch_embeddings([query])[0]
    
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=("default", query_emb),
        limit=3
    )
    
    print(f"\nKết quả tìm kiếm cho: '{query}'")
    print("-" * 60)
    for i, h in enumerate(hits, 1):
        print(f"\n{i}. Score: {h.score:.3f}")
        print(f"Tên bác sĩ: {h.payload.get('ten_bac_si', 'N/A')}")
        print(f"Chuyên môn: {', '.join(h.payload.get('chuyen_mon', []))}")
        print(f"Nơi làm việc: {h.payload.get('noi_lam_viec', 'N/A')}")
        print("-" * 60)
