from pinecone.grpc import PineconeGRPC as Pinecone
from tqdm import tqdm
from datetime import datetime
from typing import List
from loguru import logger
from generate import ImageParsing
from utils import get_transcript_dict, find_transcript
from constants import EMBED_MODEL, EMBED_TEMPLATE, EXTRACT_PROMPT
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
        self.model_vlm = ImageParsing(
            base_url="http://10.36.16.97:11434/api/generate"
        )
        self.client_db = Pinecone(
            api_key=os.getenv("MINDFLIX_PINECONE")
        )
        self.index = self.client_db.Index(f"{self.db_name}")

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
            type="search_query",
            content=f"{input_text}"
        )

        response = self.index.query(
            namespace=video_id,
            vector=query_embedding[0],
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

        logger.debug(str(img_dir_fpath[0].split('/')[-1]).split('_')[1])

        video_id = str(img_dir_fpath[0].split('/')[-1]).split('_')[1]
        logger.debug(f"VIDEO_ID: {video_id}")

        vectors = []

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

            embedding = self.create_embedding_unit(
                type="search_document",
                content=prompt
            )

            logger.debug(embedding[0].shape)

            timestamp_yt = time_obj.strftime("%Hh%Mm%Ss")

            payload = {
                "id": f"id-{video_id}_{timestamp}",
                "values": embedding[0].tolist(),
                "metadata": {
                    "yt_url": f"{yt_video.format(video_id)}",
                    "desc": f"{prompt}",
                    "timestamp": f"{yt_video.format(video_id)}&t={timestamp_yt}"
                }
            }
            logger.debug(payload)
            vectors.append(payload)
        
        self.index.upsert(
            vectors=vectors,
            namespace=video_id
        )
        return len(vectors)

if __name__ == "__main__":
    obj = VectorDB(
        db_name="mindflix-nomic",
        embed_model=f"{EMBED_MODEL}"
    )
    print(obj.query_db(
        "पाती है आमतौर पर एक छोटी वीडियो बनाने के",
        "ftDsSB3F5kg"
    ))
    # obj.push_desc_embedding(
    #     vid_dir="./run_1",
    #     transcript_path="./data/ftDsSB3F5kg/ftDsSB3F5kg_captions.txt"
    # )