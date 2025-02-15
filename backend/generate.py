from ultralytics import YOLO
from tqdm import tqdm
import requests
import os
import base64

class ImageParsing:
    def __init__(
        self, 
        yolo_version: str, 
        vlm_avail: str = "llava:7b-v1.5-q6_K"
    ):
        self.yolo_model = YOLO(
            f"{yolo_version}n.pt"
        )
        self.vlm = vlm_avail

    def convert_image(self, img_file):
        with open(img_file, "rb") as image:
            return base64.b64encode(image.read()).decode('utf-8')

    def yolo_inference_on_yt_video(self, vid_dir: str, save_dir: str):
        img_dir = os.listdir(vid_dir)
        img_dir_fpath = [f"{vid_dir}/{img}" for img in img_dir]
        os.makedirs(save_dir, exist_ok=True)

        results = self.yolo_model(
            img_dir_fpath, 
            stream=True, 
            save=True, 
            project=save_dir,
            exist_ok=True,
            conf=0.4
        )

        for i, result in enumerate(tqdm(results)):
            video_id = img_dir[i].split('_')[0]
            timestamp_jpg = img_dir[i].split('_')[-1]
            result.save(filename=f"{save_dir}result_{video_id}_{timestamp_jpg}")

    def create_scene_description(self, prompt: str, img_file: str):
        base64_image = self.convert_image(img_file=img_file)

        payload = {
            "model": f"{self.vlm}",
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
    # img_parse = ImageParsing(
    #     "yolo11"
    # )
    # img_parse.yolo_inference_on_yt_video(
    #     vid_dir="./data/ftDsSB3F5kg",
    #     save_dir="./run_1/"
    # )