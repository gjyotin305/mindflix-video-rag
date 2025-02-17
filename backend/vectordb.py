from pinecone.grpc import PineconeGRPC as Pinecone
from tqdm import tqdm
from datetime import datetime
from typing import List
from loguru import logger
import requests
from .generate import ImageParsing
from .utils import get_transcript_dict, find_transcript
from .constants import EMBED_TEMPLATE, EXTRACT_PROMPT
from sentence_transformers import SentenceTransformer
import os

class VectorDB:
    def __init__(self, db_name: str, embed_model: str, local_bool: bool = False):
        self.db_name = db_name
        self.embed_model = embed_model
        self.local_bool = local_bool
        if self.local_bool:
            self.model = SentenceTransformer(
                f"{self.embed_model}",
                trust_remote_code=True
            )
        else:
            self.model = "snowflake-arctic-embed2"
            self.base_url = os.getenv("MINDFLIX_EMBED_URL")

        self.model_vlm = ImageParsing(
            base_url=os.getenv("MINDFLIX_BASE_URL")
        )
        self.client_db = Pinecone(
            api_key=os.getenv("MINDFLIX_PINECONE")
        )
        self.index = self.client_db.Index(f"{self.db_name}")

    def create_embedding_host(self, type: str, content: str):
        if type == "query":
            payload = {
                "model": f"{self.model}",
                "input": f"{type}: {content}",
            }
        else:
            payload = {
                "model": f"{self.model}",
                "input": f"{type}: {content}",
            }
        headers = {
            'Accept-Encoding': 'gzip', 
            'Content-Type': 'application/json'
        }
        response = requests.post(
            url=self.base_url,
            json=payload,
            headers=headers
        )

        return response.json()['embeddings']

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

    def query_db(self, input_text: str, video_id: str, top_k: int = 3):
        query_embedding = self.create_embedding_unit(
            type="query",
            content=f"{input_text}"
        )

        logger.debug("Embedding Generated | Retrieving from vectorDB")
        
        response = self.index.query(
            namespace=video_id,
            vector=query_embedding[0][:768],
            top_k=top_k,
            include_metadata=True,
            include_values=False
        ) 
        
        return response['matches']


    def push_desc_embedding(
        self, 
        vid_dir: str, 
        transcript_path: str
    ):
        assert os.path.exists(vid_dir)
        img_dir = os.listdir(vid_dir)   
        img_dir_fpath = [f"{vid_dir}/{img}" for img in img_dir]  
        
        srt_dict = get_transcript_dict(
            transcript_path=transcript_path
        )

        yt_video = "https://youtube.com/watch?v={}"

        video_id = str(img_dir_fpath[0].split('/')[-1]).split('_')[1]
        logger.debug(f"VIDEO_ID: {video_id}")

        vectors = []
        count = 0
        for index, object in enumerate(tqdm(img_dir_fpath)):
            timestamp = str(object.split('_')[-1]).split('.')[0]
            time_obj = datetime.strptime(timestamp, "%H:%M:%S").time()
            
            ctr = find_transcript(
                time_obj,
                srt_dict
            )

            transcript = ""

            for index in ctr:
                transcript += srt_dict[index]['text']
                transcript += "\n"
            
            scene_desc = self.model_vlm.create_scene_description(
                prompt=f"{EXTRACT_PROMPT}",
                img_file=object
            )

            prompt = EMBED_TEMPLATE.format(transcript, scene_desc)

            embedding = self.create_embedding_host(
                type="search_document",
                content=prompt
            )

            timestamp_yt = time_obj.strftime("%Hh%Mm%Ss")

            payload = {
                "id": f"id-{video_id}_{timestamp}",
                "values": embedding[0][:768],
                "metadata": {
                    "yt_url": f"{yt_video.format(video_id)}",
                    "desc": f"{prompt}",
                    "timestamp": f"{yt_video.format(video_id)}&t={timestamp_yt}"
                }
            }

            vectors.append(payload)
            count += 1
            logger.debug(f"VECTORS ADDED {count}/{len(img_dir_fpath)}")

        self.index.upsert(
            vectors=vectors,
            namespace=video_id
        )
        logger.debug(f"SENT {len(vectors)} TO VECTOR DB")
        return len(vectors)