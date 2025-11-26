"""
Update Memory Module - Quản lý long-term memory
Chức năng:
1. Add data từ longterm_temp.txt vào Qdrant 
2. Tóm tắt nội dung longterm_temp.txt bằng LLM và lưu vào longterm.txt
3. Quản lý longterm.txt: giữ tối đa 10 ngày gần nhất
4. Quản lý Qdrant: giữ tối đa 10 ngày gần nhất
"""

import os
import sys
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from protonx import ProtonX
from openai import OpenAI

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    PROTONX_API_KEY,
    LONGTERM_COLLECTION_NAME,
    DEFAULT_LONGTERM_FILE,
    OPENAI_API_KEY,
)

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryUpdater:
    """Class để quản lý việc cập nhật và dọn dẹp long-term memory"""
    
    def __init__(
        self,
        openai_api_key: str,
        longterm_file: str = DEFAULT_LONGTERM_FILE,
        longterm_temp_file: str = "longterm_temp.txt",
        max_days: int = 10,
        model_name: str = "gpt-4o-mini"
    ):
        """
        Khởi tạo MemoryUpdater
        
        Args:
            openai_api_key: OpenAI API key để sử dụng LLM
            longterm_file: File lưu long-term memory chính (mặc định: longterm.txt)
            longterm_temp_file: File tạm chứa memory trong ngày (mặc định: longterm_temp.txt)
            max_days: Số ngày tối đa giữ lại (mặc định: 10)
            model_name: Model OpenAI để tóm tắt (mặc định: gpt-4o-mini)
        """
        self.openai_api_key = openai_api_key
        self.longterm_file = longterm_file
        self.longterm_temp_file = longterm_temp_file
        self.max_days = max_days
        self.model_name = model_name
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.protonx_client = ProtonX(api_key=PROTONX_API_KEY)
        
        logger.info("✅ MemoryUpdater initialized successfully")
    
    def _read_temp_file(self) -> str:
        """Đọc nội dung từ file longterm_temp.txt"""
        if not os.path.exists(self.longterm_temp_file):
            logger.warning(f"⚠️ File {self.longterm_temp_file} không tồn tại")
            return ""
        
        with open(self.longterm_temp_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        logger.info(f"📖 Đọc {len(content)} ký tự từ {self.longterm_temp_file}")
        return content
    
    def _summarize_with_llm(self, content: str) -> str:
        """
        Sử dụng LLM để tóm tắt nội dung trong ngày
        
        Args:
            content: Nội dung cần tóm tắt
            
        Returns:
            Nội dung đã được tóm tắt
        """
        if not content:
            return ""
        
        prompt = f"""
Bạn là một trợ lý thông minh. Nhiệm vụ của bạn là tóm tắt các hoạt động và thông tin cá nhân của người dùng trong ngày.

THÔNG TIN TRONG NGÀY:
{content}

Hãy tóm tắt lại thành một đoạn văn ngắn gọn (khoảng 2-3 câu)

TÓM TẮT (chỉ trả về nội dung tóm tắt, không thêm tiêu đề hay giải thích):
"""
        
        try:
            logger.info("🤖 Đang tóm tắt nội dung bằng LLM...")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"✅ Tóm tắt thành công (tokens: {response.usage.total_tokens})")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tóm tắt: {str(e)}")
            # Fallback: trả về nội dung gốc nếu LLM fail
            return content
    
    def _add_temp_to_qdrant(self) -> int:
        """
        Thêm các entry từ longterm_temp.txt vào Qdrant
        
        Returns:
            Số lượng entry đã thêm
        """
        content = self._read_temp_file()
        if not content:
            logger.warning("⚠️ Không có nội dung để thêm vào Qdrant")
            return 0
        
        # Parse các dòng từ file temp
        entries = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse format: [yyyy-mm-dd HH:MM] text
            match = re.match(r'\[([\d\-]+\s+[\d:]+)\]\s*(.*)', line)
            if match:
                timestamp_str = match.group(1)
                text = match.group(2).strip()
                
                # Extract date (yyyy-mm-dd)
                date_match = re.match(r'([\d\-]+)', timestamp_str)
                date_str = date_match.group(1) if date_match else timestamp_str
                
                entries.append({
                    'timestamp': timestamp_str,
                    'date': date_str,
                    'text': line,
                    'text_without_timestamp': text
                })
            else:
                # Không có timestamp, dùng ngày hiện tại
                today = datetime.now().strftime("%Y-%m-%d")
                entries.append({
                    'timestamp': today,
                    'date': today,
                    'text': line,
                    'text_without_timestamp': line
                })
        
        if not entries:
            logger.warning("⚠️ Không parse được entry nào từ file temp")
            return 0
        
        logger.info(f"📝 Đã parse {len(entries)} entries từ file temp")
        
        try:
            # Get current point count to generate new IDs
            collection_info = self.qdrant_client.get_collection(LONGTERM_COLLECTION_NAME)
            next_id = collection_info.points_count + 1
            
            # Create embeddings
            texts = [entry['text_without_timestamp'] for entry in entries]
            logger.info(f"🔤 Đang tạo embeddings cho {len(texts)} entries...")
            
            response = self.protonx_client.embeddings.create(texts)
            if isinstance(response, dict):
                embeddings = [item["embedding"] for item in response["data"]]
            else:
                embeddings = [item.embedding for item in response.data]
            
            logger.info(f"✅ Đã tạo {len(embeddings)} embeddings")
            
            # Create points
            points = []
            for i, (entry, embedding) in enumerate(zip(entries, embeddings)):
                point = PointStruct(
                    id=next_id + i,
                    vector={"default": embedding},
                    payload={
                        "text": entry['text'],
                        "text_without_timestamp": entry['text_without_timestamp'],
                        "timestamp": entry['timestamp'],
                        "date": entry['date'],  # Thêm field date để dễ filter
                        "created_at": datetime.now().isoformat()
                    }
                )
                points.append(point)
            
            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=LONGTERM_COLLECTION_NAME,
                points=points
            )
            
            logger.info(f"✅ Đã thêm {len(points)} points vào Qdrant")
            return len(points)
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi thêm vào Qdrant: {str(e)}")
            return 0
    
    def _save_summary_to_longterm(self, summary: str, date: str):
        """
        Lưu tóm tắt vào file longterm.txt với format [yyyy-mm-dd]
        
        Args:
            summary: Nội dung tóm tắt
            date: Ngày theo format yyyy-mm-dd
        """
        if not summary:
            logger.warning("⚠️ Không có nội dung tóm tắt để lưu")
            return
        
        # Format: [yyyy-mm-dd HH:MM:SS] summary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {summary.strip()}\n"
        
        with open(self.longterm_file, 'a', encoding='utf-8') as f:
            f.write(line)
        
        logger.info(f"✅ Đã lưu tóm tắt vào {self.longterm_file}")
    
    def _parse_longterm_file(self) -> List[Dict[str, str]]:
        """
        Parse file longterm.txt và trích xuất các entry
        
        Returns:
            List các entry với thông tin timestamp và date
        """
        if not os.path.exists(self.longterm_file):
            logger.warning(f"⚠️ File {self.longterm_file} không tồn tại")
            return []
        
        entries = []
        with open(self.longterm_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: [yyyy-mm-dd HH:MM:SS] text
                match = re.match(r'\[([\d\-]+\s+[\d:]+)\]\s*(.*)', line)
                if match:
                    timestamp_str = match.group(1)
                    text = match.group(2).strip()
                    
                    # Extract date (yyyy-mm-dd)
                    date_match = re.match(r'([\d\-]+)', timestamp_str)
                    date_str = date_match.group(1) if date_match else timestamp_str
                    
                    entries.append({
                        'line_num': line_num,
                        'timestamp': timestamp_str,
                        'date': date_str,
                        'text': text,
                        'full_line': line
                    })
        
        logger.info(f"📖 Đã parse {len(entries)} entries từ {self.longterm_file}")
        return entries
    
    def _cleanup_longterm_file(self):
        """
        Dọn dẹp file longterm.txt: giữ lại tối đa max_days ngày gần nhất
        """
        entries = self._parse_longterm_file()
        if not entries:
            logger.info("ℹ️ File longterm.txt trống, không cần dọn dẹp")
            return
        
        # Group by date
        date_groups = defaultdict(list)
        for entry in entries:
            date_groups[entry['date']].append(entry)
        
        unique_dates = sorted(date_groups.keys(), reverse=True)  # Mới nhất trước
        
        logger.info(f"📊 Tìm thấy {len(unique_dates)} ngày khác nhau trong file")
        
        if len(unique_dates) <= self.max_days:
            logger.info(f"✅ Số ngày ({len(unique_dates)}) <= {self.max_days}, không cần xoá")
            return
        
        # Giữ lại max_days ngày gần nhất
        dates_to_keep = set(unique_dates[:self.max_days])
        dates_to_remove = set(unique_dates[self.max_days:])
        
        logger.info(f"🗑️ Sẽ xoá {len(dates_to_remove)} ngày cũ: {sorted(dates_to_remove)}")
        
        # Filter entries to keep
        entries_to_keep = [
            entry for entry in entries
            if entry['date'] in dates_to_keep
        ]
        
        # Rewrite file
        with open(self.longterm_file, 'w', encoding='utf-8') as f:
            for entry in entries_to_keep:
                f.write(entry['full_line'] + '\n')
        
        logger.info(f"✅ Đã dọn dẹp file, giữ lại {len(entries_to_keep)} entries từ {len(dates_to_keep)} ngày")
    
    def _get_all_dates_in_qdrant(self) -> List[str]:
        """
        Lấy danh sách tất cả các ngày (date) có trong Qdrant
        
        Returns:
            List các ngày duy nhất (sorted)
        """
        try:
            # Scroll through all points to get dates
            offset = None
            all_dates = set()
            
            while True:
                # Scroll with limit
                result = self.qdrant_client.scroll(
                    collection_name=LONGTERM_COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_offset = result
                
                if not points:
                    break
                
                # Extract dates
                for point in points:
                    date = point.payload.get('date')
                    if date:
                        all_dates.add(date)
                    else:
                        # Fallback: extract from timestamp
                        timestamp = point.payload.get('timestamp', '')
                        date_match = re.match(r'([\d\-]+)', timestamp)
                        if date_match:
                            all_dates.add(date_match.group(1))
                
                # Check if we've reached the end
                if next_offset is None:
                    break
                
                offset = next_offset
            
            dates_list = sorted(list(all_dates), reverse=True)  # Mới nhất trước
            logger.info(f"📊 Tìm thấy {len(dates_list)} ngày khác nhau trong Qdrant")
            return dates_list
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi lấy danh sách ngày từ Qdrant: {str(e)}")
            return []
    
    def _cleanup_qdrant(self):
        """
        Dọn dẹp Qdrant: giữ lại tối đa max_days ngày gần nhất
        """
        all_dates = self._get_all_dates_in_qdrant()
        
        if not all_dates:
            logger.info("ℹ️ Không tìm thấy ngày nào trong Qdrant")
            return
        
        if len(all_dates) <= self.max_days:
            logger.info(f"✅ Số ngày ({len(all_dates)}) <= {self.max_days}, không cần xoá")
            return
        
        # Dates to remove (older than max_days)
        dates_to_remove = all_dates[self.max_days:]
        
        logger.info(f"🗑️ Sẽ xoá {len(dates_to_remove)} ngày cũ từ Qdrant: {dates_to_remove}")
        
        # Delete points by date
        try:
            for date in dates_to_remove:
                # Scroll and collect point IDs for this date
                point_ids_to_delete = []
                offset = None
                
                while True:
                    result = self.qdrant_client.scroll(
                        collection_name=LONGTERM_COLLECTION_NAME,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    points, next_offset = result
                    
                    if not points:
                        break
                    
                    # Collect IDs for this date
                    for point in points:
                        point_date = point.payload.get('date')
                        if not point_date:
                            # Fallback: extract from timestamp
                            timestamp = point.payload.get('timestamp', '')
                            date_match = re.match(r'([\d\-]+)', timestamp)
                            if date_match:
                                point_date = date_match.group(1)
                        
                        if point_date == date:
                            point_ids_to_delete.append(point.id)
                    
                    if next_offset is None:
                        break
                    
                    offset = next_offset
                
                # Delete collected points
                if point_ids_to_delete:
                    self.qdrant_client.delete(
                        collection_name=LONGTERM_COLLECTION_NAME,
                        points_selector=point_ids_to_delete
                    )
                    logger.info(f"🗑️ Đã xoá {len(point_ids_to_delete)} points từ ngày {date}")
            
            logger.info(f"✅ Đã dọn dẹp Qdrant, giữ lại {self.max_days} ngày gần nhất")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi dọn dẹp Qdrant: {str(e)}")

    def _clear_temp_file(self):
        """
        Xoá nội dung trong file longterm_temp.txt
        """
        if os.path.exists(self.longterm_temp_file):
            with open(self.longterm_temp_file, 'w', encoding='utf-8') as f:
                f.write('')
            logger.info(f"✅ Đã xoá nội dung trong file {self.longterm_temp_file}")
            return True
        else:
            logger.warning(f"⚠️ File {self.longterm_temp_file} không tồn tại")
            return False
    
    def update_memory(self, date: Optional[str] = None) -> Dict[str, any]:
        """
        Hàm chính để cập nhật memory
        
        Quy trình:
        - Tóm tắt nội dung longterm_temp.txt bằng LLM
        - Lưu tóm tắt vào longterm.txt
        - Dọn dẹp longterm.txt (giữ max_days ngày)
        - Dọn dẹp Qdrant (giữ max_days ngày)
        
        Args:
            date: Ngày để lưu tóm tắt (format: yyyy-mm-dd). Nếu None, dùng ngày hiện tại
            
        Returns:
            Dict chứa thông tin về quá trình cập nhật
        """
        logger.info("=" * 80)
        logger.info("🚀 BẮT ĐẦU CẬP NHẬT MEMORY")
        logger.info("=" * 80)
        
        result = {
            'success': True,
            'date': date or datetime.now().strftime("%Y-%m-%d"),
            'qdrant_added': 0,
            'summary': '',
            'longterm_cleaned': False,
            'qdrant_cleaned': False,
            'errors': []
        }
        
        try:
            logger.info("\n🤖 BƯỚC 1: Tóm tắt nội dung bằng LLM")
            logger.info("-" * 80)
            temp_content = self._read_temp_file()
            if temp_content:
                summary = self._summarize_with_llm(temp_content)
                result['summary'] = summary
                
                # Step 3: Save to longterm.txt
                logger.info("\n💾 BƯỚC 2: Lưu tóm tắt vào longterm.txt")
                logger.info("-" * 80)
                self._save_summary_to_longterm(summary, result['date'])
            else:
                logger.warning("⚠️ Không có nội dung để tóm tắt")
            
            # Step 4: Cleanup longterm.txt
            logger.info("\n🧹 BƯỚC 3: Dọn dẹp longterm.txt")
            logger.info("-" * 80)
            self._cleanup_longterm_file()
            result['longterm_cleaned'] = True
            
            # Step 5: Cleanup Qdrant
            logger.info("\n🧹 BƯỚC 4: Dọn dẹp Qdrant")
            logger.info("-" * 80)
            self._cleanup_qdrant()
            result['qdrant_cleaned'] = True

            # Step 6: Clear temp file
            logger.info("\n🧹 BƯỚC 5: Xoá temp file")
            logger.info("-" * 80)
            self._clear_temp_file()
            result['temp_file_cleared'] = True

            
            logger.info("\n" + "=" * 80)
            logger.info("✅ CẬP NHẬT MEMORY HOÀN TẤT")
            logger.info("=" * 80)
            logger.info(f"📊 Tóm tắt:")
            logger.info(f"  - Đã thêm {result['qdrant_added']} entries vào Qdrant")
            logger.info(f"  - Đã tóm tắt và lưu vào longterm.txt")
            logger.info(f"  - Đã dọn dẹp file và database (giữ {self.max_days} ngày)")
            
        except Exception as e:
            logger.error(f"❌ LỖI: {str(e)}")
            result['success'] = False
            result['errors'].append(str(e))
            import traceback
            logger.error(traceback.format_exc())
        
        return result


def update_memory(
    openai_api_key: str,
    longterm_file: str = DEFAULT_LONGTERM_FILE,
    longterm_temp_file: str = "longterm_temp.txt",
    max_days: int = 10,
    model_name: str = "gpt-4o-mini",
    date: Optional[str] = None
) -> Dict[str, any]:
    """
    Hàm tiện ích để cập nhật memory (wrapper function)
    
    Args:
        openai_api_key: OpenAI API key
        longterm_file: File lưu long-term memory chính
        longterm_temp_file: File tạm chứa memory trong ngày
        max_days: Số ngày tối đa giữ lại (mặc định: 10)
        model_name: Model OpenAI để tóm tắt
        date: Ngày để lưu tóm tắt (format: yyyy-mm-dd)
        
    Returns:
        Dict chứa thông tin về quá trình cập nhật
    """
    updater = MemoryUpdater(
        openai_api_key=openai_api_key,
        longterm_file=longterm_file,
        longterm_temp_file=longterm_temp_file,
        max_days=max_days,
        model_name=model_name
    )
    
    return updater.update_memory(date=date)


# ============================================================================
# Main - For Testing
# ============================================================================

def main():
    """Main function for testing"""
    # Run update
    result = update_memory(
        openai_api_key=OPENAI_API_KEY,
        max_days=10
    )
    
    # Print result
    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ CẬP NHẬT")
    print("=" * 80)
    print(f"Success: {result['success']}")
    print(f"Date: {result['date']}")
    print(f"Qdrant added: {result['qdrant_added']} entries")
    print(f"Summary: {result['summary'][:100]}..." if result['summary'] else "Summary: (empty)")
    print(f"Longterm cleaned: {result['longterm_cleaned']}")
    print(f"Qdrant cleaned: {result['qdrant_cleaned']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

