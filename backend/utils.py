import decord
import cv2
import numpy as np
from tqdm import tqdm
import os
import re
from urllib.parse import urlparse, parse_qs
from loguru import logger
from datetime import timedelta, datetime
from pytubefix import YouTube
decord.bridge.set_bridge("native")

def find_transcript(time_obj: datetime.time, srt_dict):
    result  = []
    for index, (item) in srt_dict.items():
        obj_start, obj_end = item['start'], item['end']
        if time_obj >= datetime.strptime(obj_start, "%H:%M:%S,%f").time() and time_obj <= datetime.strptime(obj_end, "%H:%M:%S,%f").time():
            result.append(index)
    
    return result

def parse_transcript(srt_text: str):
        srt_dict = {}
    
        blocks = srt_text.strip().split('\n\n')
        for block in blocks:
            lines = block.split('\n')
            if len(lines) < 3:
                continue
            
            index = lines[0].strip()
            time_match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", lines[1])
            if not time_match:
                continue
            
            start_time, end_time = time_match.groups()
            text = ' '.join(lines[2:])
            
            srt_dict[index] = {"start": start_time, "end": end_time, "text": text}
        
        return srt_dict

def get_transcript_dict(
    transcript_path: str
):
    with open(transcript_path, "r") as f:
        transcript = f.read()
        f.close()
    
    srt_dict = parse_transcript(transcript)
    
    return srt_dict

class VideoBreakDown:
    def __init__(self, backend: str = "decord"):
        self.backend = backend

    def get_vid_transcript(
        self, 
        vid_url: str, 
        save_dir: str="./data/",
        lang: str = "hi"
    ):
        yt = YouTube(url=vid_url)
        parsed_url = urlparse(
            url=vid_url
        )
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]
        captions = yt.captions[f'a.{lang}']
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}{video_id}", exist_ok=True)

        with open(f"{save_dir}{video_id}/{video_id}_captions.txt", "w") as f:
            f.write(captions.generate_srt_captions())
            f.close()

        print(f"Saving {video_id}_captions.txt")


    def get_vid_to_frame_by_frame(
        self, 
        vid_url: str,
        save_dir: str = "./data/", 
        every: int = 30
    ):
        yt = YouTube(url=vid_url)
        parsed_url = urlparse(
            url=vid_url
        )
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]
        logger.debug(f"YT VIDEO TITLE: {yt.title}")
        yt_down = yt.streams.get_highest_resolution()
        logger.debug(f"YT VID: RESOLUTION {yt_down.resolution}")
        yt_down.download(output_path=f"{save_dir}", filename=f"{video_id}.mp4")

        output_dir = f"{save_dir}{video_id}"
        os.makedirs(output_dir, exist_ok=True)
        vr = decord.VideoReader(f"{save_dir}{video_id}.mp4")
        
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
    # vid.get_vid_to_frame_by_frame(
    #     "https://www.youtube.com/watch?v=ftDsSB3F5kg"
    # )
    # vid.get_vid_transcript(
    #     "https://www.youtube.com/watch?v=ftDsSB3F5kg"
    # )
    print(vid.get_transcript_dict(
        "./data/ftDsSB3F5kg/ftDsSB3F5kg_captions.txt"
    ))

