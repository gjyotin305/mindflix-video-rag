from pinecone.grpc import PineconeGRPC as Pinecone
from tqdm import tqdm
from PIL import Image
from typing import List
from constants import EMBED_MODEL
from sentence_transformers import SentenceTransformer
import os

class VectorDB:
    def __init__(self, db_name: str, embed_model: str):
        self.db_name = db_name
        self.embed_model = embed_model
        self.model = SentenceTransformer(
            f"{self.embed_model}",
            trust_remote_code=True
        )
        self.client_db = Pinecone(
            api_key=os.getenv("MINDFLIX_PINECONE")
        )

    def create_embedding_unit(self, type: str, content: str):
        sentences = [
            f'{type}: {content}'
        ]
        embeddings = self.model.encode(sentences)
        return embeddings

    def create_embedding_batch(self, type: str, content: List[str]):
        sentences = [f"{type}: {text}" for text in content]
        embeddings = self.model.encode(sentences)
        return embeddings

    def push_desc_embedding(self, img_dir: str):
        pass

if __name__ == "__main__":
    obj = VectorDB(
        db_name="test",
        embed_model=f"{EMBED_MODEL}"
    )
    print(len(obj.create_embedding_unit(
        type="search_document",
        content="OLLLAA MY AMIGOOO"
    )[0]))