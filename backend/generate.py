from ultralytics import YOLO
from tqdm import tqdm
import requests
import os
import base64

class ImageParsing:
    def __init__(self, yolo_version: str, vlm_avail: str = "llava"):
        self.yolo_model = YOLO(
            f"{yolo_version}n.pt"
        )
        self.vlm = "llava:7b-v1.5-q6_K"

    def convert_image(self, img_file):
        with open(img_file, "rb") as image:
            return base64.b64encode(image.read()).decode('utf-8')

    def yolo_inference_on_yt_video(self, vid_dir: str, save_dir: str):
        img_dir = os.listdir(vid_dir)
        img_dir_fpath = [f"{vid_dir}/{img}" for img in img_dir]

        results = self.yolo_model(
            img_dir_fpath, 
            stream=True, 
            save=True, 
            project=save_dir,
            exist_ok=True
        )

        for i, result in tqdm(enumerate(results)):
            result.save(filename=f"{save_dir}result_{i}.jpg")

    def create_scene_description(self, prompt: str, img_file: str):
        base64_image = self.convert_image(img_file=img_file)

        payload = {
            "model": f"{self.model}",
            "prompt": f"{prompt}",
            "stream": False,
            "images": [base64_image]
        }

        headers = {'Accept-Encoding': 'gzip'}
        response = requests.post(
            self.base_url,
            json=payload,
            headers=headers
        )

        return response.json()['response']


if __name__ == "__main__":
    img_parse = ImageParsing(
        "yolo11"
    )
    img_parse.yolo_inference_on_yt_video(
        vid_url="https://www.youtube.com/watch?v=ftDsSB3F5kg",
        save_dir="./run_1/"
    )