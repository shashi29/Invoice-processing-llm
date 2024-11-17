import json
import os
import uuid
import zlib
import logging
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, UpdateStatus, OptimizersConfigDiff
from tqdm import tqdm  # Import tqdm for progress tracking

import json
from pathlib import Path
from collections import OrderedDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_json_file(file_path):
    """
    Reads a JSON file and returns its content as a Python dictionary.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        dict: Parsed JSON content as a dictionary.
    """
    try:
        file_path = Path(file_path)
        
        # Ensure the file exists
        if not file_path.is_file():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Read and parse the JSON file
        with open(file_path, 'r', encoding='utf-8') as json_file:
            content = json.load(json_file)
        
        return content
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        raise
    

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: Any):
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

model_cache = LRUCache(capacity=1)

class EmbeddingManager:
    def __init__(self, embedding_model_name: str, quadrant_url: str):
        self.embedding_model = self.load_embedding_model(embedding_model_name)
        self.qdrant_client = QdrantClient(base_url=quadrant_url)

    def create_collection(self, collection_name: str, vector_size: int, shard_number: int = 4):
        """
        Create a collection in Qdrant with shard number and indexing disabled for initial bulk upload.
        """
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, 
                    distance=Distance.COSINE,
                    on_disk=True  # Store vectors on disk
                ),
                shard_number=shard_number,  # Split data into multiple shards
                optimizers_config=OptimizersConfigDiff(indexing_threshold=0)  # Disable indexing for faster upload
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def update_indexing(self, collection_name: str, indexing_threshold: int = 20000):
        """
        Update collection indexing after bulk upload to enable indexing.
        """
        try:
            self.qdrant_client.update_collection(
                collection_name=collection_name,
                optimizers_config=OptimizersConfigDiff(indexing_threshold=indexing_threshold)
            )
            logger.info(f"Indexing enabled with threshold {indexing_threshold} for collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error updating collection indexing: {e}")
            raise

    def check_collection_exists(self, collection_name: str) -> bool:
        """
        Check if the collection already exists in Qdrant.
        """
        try:
            collections = self.qdrant_client.get_collections()
            return any(collection.name == collection_name for collection in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            raise
                
    def load_embedding_model(self, model_name: str) -> SentenceTransformer:
        model = model_cache.get(model_name)
        if model is None:
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name, trust_remote_code=True)
            model_cache.put(model_name, model)
        return model

    def process_pdf_json(self, pdf_json: Dict[str, str], document_name: str, chunk_size: int = 500, chunk_overlap: int = 200):
        try:
            logger.info(f"Processing PDF JSON: {document_name}")
            collection_name = "contract_collection"
            # Set up text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            points = []
            for page_key, text in tqdm(pdf_json.items(), desc="Processing pages"):
                # Extract page number from the key (e.g., "page_1" -> 1)
                page_number = int(page_key.split("_")[1])

                # Split text into chunks
                chunks = text_splitter.create_documents([text])

                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    embedding = self.embedding_model.encode(chunk.page_content).tolist()
                    payload = {
                        "chunk_id":chunk_id,          
                        "text":chunk.page_content,
                        "page_number":page_number,
                        "document_name":document_name              
                    }
                    points.append(PointStruct(id=chunk_id, vector=embedding, payload=payload))

            # Insert points into the collection
            operation_info = self.qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                logger.info(f"Successfully inserted {len(points)} embeddings from '{document_name}' into collection '{collection_name}'.")
            else:
                raise Exception("Failed to insert embeddings.")
            
            logger.info(f"Successfully processed and inserted embeddings for: {document_name}")
        except Exception as ex:
            logger.error(f"Error processing PDF JSON for {document_name}: {ex}")


# Main function
def process_pdf_document(embedding_manager: EmbeddingManager, pdf_json: Dict[str, str], document_name: str):
    embedding_manager.process_pdf_json(pdf_json, document_name)

if __name__ == "__main__":
    # Initialize EmbeddingManager with Quadrant connection details
    embedding_manager = EmbeddingManager(
        embedding_model_name="mixedbread-ai/mxbai-embed-large-v1",
        quadrant_url="localhost:6333"
    )

    # Check if collection exists, otherwise create it with shard number and indexing disabled
    embedding_dim = 1024  # Replace this with the actual embedding dimension
    collection_name = "contract_collection"
    if not embedding_manager.check_collection_exists(collection_name):
        embedding_manager.create_collection(collection_name, vector_size=embedding_dim)
        
    file_path = "/root/code/Invoice-processing-llm/PetroChoice_SAIA_LTL_Exhibit E_ 08.22.2022 (1) (1)/PetroChoice_SAIA_LTL_Exhibit E_ 08.22.2022 (1) (1)_text.json"
    document_name =   file_path.split("/")[-1]  
    pdf_json = read_json_file(file_path)

    # Process the JSON and insert embeddings into Quadrant
    process_pdf_document(embedding_manager, pdf_json, document_name)
