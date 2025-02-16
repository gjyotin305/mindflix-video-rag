from ultralytics import YOLO
from tqdm import tqdm
import requests
from loguru import logger
import os
from .constants import EXTRACT_PROMPT
import base64

class ImageParsing:
    def __init__(
        self,
        base_url: str, 
        yolo_version: str = "yolo11", 
        vlm_avail: str = "llava:7b-v1.5-q6_K",
    ):
        self.yolo_model = YOLO(
            f"{yolo_version}n.pt"
        )
        self.base_url = base_url
        self.vlm = vlm_avail

    def convert_image(self, img_file):
        with open(img_file, "rb") as image:
            return base64.b64encode(image.read()).decode('utf-8')

    def yolo_inference_on_yt_video(self, vid_dir: str, save_dir: str):
        img_dir = os.listdir(vid_dir)
        img_dir_fpath = [f"{vid_dir}/{img}" for img in img_dir if not img.endswith('.txt')]
        os.makedirs(save_dir, exist_ok=True)

        results = self.yolo_model(
            img_dir_fpath, 
            stream=True,
            device="cpu",
            conf=0.4
        )

        for i, result in enumerate(tqdm(results)):
            video_id = img_dir_fpath[i].split('_')[0]
            timestamp_jpg = img_dir_fpath[i].split('_')[-1]
            result.save(filename=f"{save_dir}/result_{video_id}_{timestamp_jpg}")

    def create_scene_description(self, prompt: str, img_file: str):
        base64_image = self.convert_image(img_file=img_file)

        payload = {
            "model": f"{self.vlm}",
            "prompt": f"{prompt}",
            "stream": False,
            "images": [base64_image]
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

        return response.json()['response']


# if __name__ == "__main__":
#     img_parse = ImageParsing(
#         "http://10.36.16.97:8443/api/generate",
#         "yolo11"
#     )
#     # desc = img_parse.create_scene_description(
#     #     EXTRACT_PROMPT,
#     #     "./run_1/result_ftDsSB3F5kg_0:00:40.jpg"
#     # )
#     # print(desc)
#     img_parse.yolo_inference_on_yt_video(
#         vid_dir="./data/ftDsSB3F5kg",
#         save_dir="./run_1/"
#     )