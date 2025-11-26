import json

def filter_unique_doctors_by_specialty(input_file, output_file):
    """
    Lọc bác sĩ theo chuyên môn, chỉ giữ lại 1 bác sĩ cho mỗi tổ hợp chuyên môn duy nhất
    
    Args:
        input_file: Đường dẫn file JSON đầu vào
        output_file: Đường dẫn file JSON đầu ra
    """
    # Đọc dữ liệu từ file JSON
    print(f"📖 Đọc dữ liệu từ {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        doctors = json.load(f)
    
    print(f"📊 Tổng số bác sĩ ban đầu: {len(doctors)}")
    
    # Dictionary để lưu bác sĩ đầu tiên cho mỗi chuyên môn
    unique_doctors = {}
    specialty_key_to_doctor = {}
    
    # Lọc bác sĩ
    for doctor in doctors:
        # Lấy danh sách chuyên môn
        chuyen_mon = doctor.get("chuyen_mon", [])
        
        # Tạo key từ chuyên môn (sắp xếp để đảm bảo thứ tự không ảnh hưởng)
        # VD: ["Bác sĩ Nội khoa", "Bác sĩ Tim mạch"] = "Bác sĩ Nội khoa|Bác sĩ Tim mạch"
        if chuyen_mon:
            specialty_key = "|".join(sorted(chuyen_mon))
        else:
            specialty_key = "KHONG_CO_CHUYEN_MON"
        
        # Chỉ giữ bác sĩ đầu tiên cho mỗi chuyên môn
        if specialty_key not in specialty_key_to_doctor:
            specialty_key_to_doctor[specialty_key] = doctor
            unique_doctors[specialty_key] = {
                "chuyen_mon": chuyen_mon,
                "bac_si": doctor.get("ten_bac_si", "Unknown"),
                "count": 1
            }
        else:
            unique_doctors[specialty_key]["count"] += 1
    
    # Chuyển về list
    filtered_doctors = list(specialty_key_to_doctor.values())
    
    print(f"\n✅ Kết quả lọc:")
    print(f"   - Số bác sĩ sau khi lọc: {len(filtered_doctors)}")
    print(f"   - Số bác sĩ bị loại: {len(doctors) - len(filtered_doctors)}")
    print(f"   - Số chuyên môn duy nhất: {len(unique_doctors)}")
    
    # Hiển thị thống kê
    print(f"\n📋 Thống kê các chuyên môn:")
    print("-" * 80)
    
    # Sắp xếp theo số lượng bác sĩ bị loại (count - 1)
    sorted_specialties = sorted(unique_doctors.items(), key=lambda x: x[1]["count"], reverse=True)
    
    for idx, (key, info) in enumerate(sorted_specialties[:20], 1):  # Hiển thị top 20
        chuyen_mon_str = ", ".join(info["chuyen_mon"]) if info["chuyen_mon"] else "Không có chuyên môn"
        print(f"{idx:2d}. {chuyen_mon_str}")
        print(f"    Giữ lại: {info['bac_si']}")
        print(f"    Số bác sĩ có cùng chuyên môn: {info['count']}")
        print()
    
    if len(sorted_specialties) > 20:
        print(f"... và {len(sorted_specialties) - 20} chuyên môn khác")
    
    # Lưu vào file mới
    print(f"\n💾 Đang lưu vào {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_doctors, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Hoàn thành! Đã lưu {len(filtered_doctors)} bác sĩ vào file mới.")
    
    return filtered_doctors

if __name__ == "__main__":
    input_file = "/Users/apple/VITA /Doctor_vinmec/crawl_data/vinmec_doctors_database.json"
    output_file = "/Users/apple/VITA /Doctor_vinmec/crawl_data/vinmec_doctors_unique.json"
    
    print("=" * 80)
    print("🏥 LỌC BÁC SĨ THEO CHUYÊN MÔN DUY NHẤT")
    print("=" * 80)
    print()
    
    filtered_doctors = filter_unique_doctors_by_specialty(input_file, output_file)
    
    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH!")
    print("=" * 80)

