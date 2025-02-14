import decord
import cv2
import numpy as np
from tqdm import tqdm
import os
from urllib.parse import urlparse, parse_qs
from loguru import logger
from datetime import timedelta
from pytubefix import YouTube
decord.bridge.set_bridge("native")

class VideoBreakDown:
    def __init__(self, backend: str = "decord"):
        self.backend = backend

    def get_vid_to_frame_by_frame(self, vid_url: str, every: int = 60):
        yt = YouTube(url=vid_url)
        parsed_url = urlparse(
            url=vid_url
        )
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]
        logger.debug(f"YT VIDEO TITLE: {yt.title}")
        yt_down = yt.streams.get_highest_resolution()
        logger.debug(f"YT VID: RESOLUTION {yt_down.resolution}")
        yt_down.download(output_path=f"./data/", filename=f"{video_id}.mp4")

        output_dir = f"./data/{video_id}"
        os.makedirs(output_dir, exist_ok=True)
        vr = decord.VideoReader(f"./data/{video_id}.mp4")
        
        vr_list = list(range(0, len(vr), every))

        fps = vr.get_avg_fps()

        for index in tqdm(vr_list):
            frame = vr[index].asnumpy()
            timestamp_seconds = index/fps
            timestamp_str = str(timedelta(seconds=timestamp_seconds))
            timestamp_str = timestamp_str.split('.')[0]

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            filename = os.path.join(
                output_dir, 
                f"{video_id}_frame_{timestamp_str}.jpg"
            )
            cv2.imwrite(filename, frame_bgr)
            logger.debug(f"Saved: {filename}")

if __name__ == "__main__":
    vid = VideoBreakDown()
    vid.get_vid_to_frame_by_frame("https://www.youtube.com/watch?v=ftDsSB3F5kg")

