"""
Initialize LongTermMemory collection in Qdrant and import data from longterm.txt
"""

import os
import sys
import logging
from typing import List, Dict
from datetime import datetime
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from protonx import ProtonX
from config import QDRANT_URL, QDRANT_API_KEY, PROTONX_API_KEY

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
LONGTERM_COLLECTION = "longterm_memory"
LONGTERM_FILE = "longterm.txt"
EMBEDDING_DIMENSION = 768  # ProtonX embedding dimension


def parse_longterm_file(file_path: str) -> List[Dict[str, str]]:
    """
    Parse longterm.txt file and extract entries
    
    Format: [2024-01-15 10:30:45] Information text
    
    Returns:
        List of dicts with 'timestamp', 'text', and 'text_without_timestamp'
    """
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found")
        return []
    
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Parse format: [timestamp] text
            match = re.match(r'\[(.*?)\]\s*(.*)', line)
            if match:
                timestamp_str = match.group(1)
                text = match.group(2).strip()
                
                entries.append({
                    'id': line_num,
                    'timestamp': timestamp_str,
                    'text': line,  # Full text with timestamp
                    'text_without_timestamp': text  # Text only (for embedding)
                })
            else:
                # No timestamp format, use as-is
                entries.append({
                    'id': line_num,
                    'timestamp': 'unknown',
                    'text': line,
                    'text_without_timestamp': line
                })
    
    logger.info(f"Parsed {len(entries)} entries from {file_path}")
    return entries


def create_collection(client: QdrantClient, collection_name: str, dimension: int):
    """Create Qdrant collection for long-term memory"""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(col.name == collection_name for col in collections)
        
        if exists:
            logger.warning(f"Collection {collection_name} already exists. Deleting...")
            client.delete_collection(collection_name)
        
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "default": VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            }
        )
        logger.info(f"✅ Created collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"❌ Error creating collection: {str(e)}")
        raise


def upload_entries(
    client: QdrantClient,
    protonx_client: ProtonX,
    collection_name: str,
    entries: List[Dict[str, str]]
):
    """Upload entries to Qdrant with embeddings"""
    if not entries:
        logger.warning("No entries to upload")
        return
    
    logger.info(f"Creating embeddings for {len(entries)} entries...")
    
    # Extract texts for embedding (without timestamp)
    texts = [entry['text_without_timestamp'] for entry in entries]
    
    try:
        # Create embeddings using ProtonX
        response = protonx_client.embeddings.create(texts)
        
        if isinstance(response, dict):
            embeddings = [item["embedding"] for item in response["data"]]
        else:
            embeddings = [item.embedding for item in response.data]
        
        logger.info(f"✅ Created {len(embeddings)} embeddings")
        
        # Prepare points for upload
        points = []
        for i, (entry, embedding) in enumerate(zip(entries, embeddings)):
            point = PointStruct(
                id=entry['id'],
                vector={"default": embedding},
                payload={
                    "text": entry['text'],  # Full text with timestamp
                    "text_without_timestamp": entry['text_without_timestamp'],
                    "timestamp": entry['timestamp'],
                    "created_at": datetime.now().isoformat()
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"✅ Uploaded {len(points)} points to {collection_name}")
        
        # Verify
        collection_info = client.get_collection(collection_name)
        logger.info(f"📊 Collection info: {collection_info.points_count} points")
        
    except Exception as e:
        logger.error(f"❌ Error uploading entries: {str(e)}")
        raise


def main():
    """Main function to initialize long-term memory collection"""
    logger.info("🚀 Starting LongTermMemory collection initialization...")
    
    try:
        # Initialize clients
        logger.info("Connecting to Qdrant...")
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        logger.info("Connecting to ProtonX...")
        protonx_client = ProtonX(api_key=PROTONX_API_KEY)
        
        # Parse longterm.txt
        logger.info(f"Parsing {LONGTERM_FILE}...")
        entries = parse_longterm_file(LONGTERM_FILE)
        
        if not entries:
            logger.error("❌ No entries found in longterm.txt")
            return
        
        # Create collection
        logger.info(f"Creating collection {LONGTERM_COLLECTION}...")
        create_collection(qdrant_client, LONGTERM_COLLECTION, EMBEDDING_DIMENSION)
        
        # Upload entries
        logger.info("Uploading entries to Qdrant...")
        upload_entries(qdrant_client, protonx_client, LONGTERM_COLLECTION, entries)
        
        logger.info("✅ LongTermMemory collection initialized successfully!")
        logger.info(f"📊 Total entries: {len(entries)}")
        
        # Test search
        logger.info("\n🔍 Testing search...")
        test_query = "thông tin về người dùng"
        logger.info(f"Query: {test_query}")
        
        response = protonx_client.embeddings.create([test_query])
        if isinstance(response, dict):
            query_emb = response["data"][0]["embedding"]
        else:
            query_emb = response.data[0].embedding
        
        results = qdrant_client.search(
            collection_name=LONGTERM_COLLECTION,
            query_vector=("default", query_emb),
            limit=3
        )
        
        logger.info(f"Found {len(results)} results:")
        for i, hit in enumerate(results, 1):
            logger.info(f"{i}. Score: {hit.score:.4f}")
            logger.info(f"   Text: {hit.payload['text_without_timestamp'][:100]}...")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

