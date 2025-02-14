import decord
import cv2
import numpy as np
import os
from loguru import logger
from datetime import datetime
from pytubefix import YouTube


class VideoBreakDown:
    def __init__(self, backend: str = "decord"):
        self.backend = backend

    def get_vid_to_frame_by_frame(self, vid_url: str):
        yt = YouTube(url=vid_url)
        logger.debug(f"YT VIDEO TITLE: {yt.title}")
        yt_down = yt.streams.get_highest_resolution()
        logger.debug(f"YT VID: RESOLUTION {yt_down.resolution}")
        yt_down.download()


if __name__ == "__main__":
    vid = VideoBreakDown()
    vid.get_vid_to_frame_by_frame("https://www.youtube.com/watch?v=ftDsSB3F5kg")

