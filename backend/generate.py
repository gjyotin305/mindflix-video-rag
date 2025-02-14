from ultralytics import YOLO
from tqdm import tqdm
import os

# # Load a pretrained YOLO11n model
# model = YOLO("yolo11n.pt")

# # Define source as YouTube video URL
# source = "https://youtu.be/LNwODJXcvt4"

# # Run inference on the source
# results = model(source, stream=True) 

class ImageParsing:
    def __init__(self, yolo_version: str):
        self.yolo_model = YOLO(
            f"{yolo_version}n.pt"
        )

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

    def create_scene_description(self):
        pass


if __name__ == "__main__":
    img_parse = ImageParsing(
        "yolo11"
    )
    img_parse.yolo_inference_on_yt_video(
        vid_url="https://www.youtube.com/watch?v=ftDsSB3F5kg",
        save_dir="./run_1/"
    )