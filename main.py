import gradio as gr
from backend.utils import VideoBreakDown
from backend.generate import ImageParsing
from backend.vectordb import VectorDB
from loguru import logger
from urllib.parse import parse_qs, urlparse
from backend.constants import EMBED_MODEL
import time
import os

# Function to handle video processing
def process_video(url, progress=gr.Progress(track_tqdm=True)):
    if not url.strip():
        return "⚠️ Please enter a valid video URL."

    logger.debug(f"Video BreakDown has Started: URL {url}")
    progress(0)
    vbd = VideoBreakDown()
    vbd.get_vid_to_frame_by_frame(
        vid_url=url
    )
    logger.debug("VIDEO BROKEN INTO FRAMES")
    parsed_url = urlparse(
        url=url
    )
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    vbd.get_vid_transcript(
        vid_url=url
    )
    logger.debug("VIDEO Transcript saved")

    progress(0.20)
    img_parse = ImageParsing(
        base_url=os.getenv("MINDFLIX_BASE_URL")
    )
    img_parse.yolo_inference_on_yt_video(
        vid_dir=f"./data/{video_id}",
        save_dir=f"./data/{video_id}_yolo"
    )

    progress(0.40)
    logger.debug("OBJECT DETECTION DONE")

    vdb = VectorDB(
        db_name="mindflix-nomic",
        embed_model=EMBED_MODEL
    )
    vectors = vdb.push_desc_embedding(
        vid_dir=f"./data/{video_id}_yolo",
        transcript_path=f"./data/{video_id}/{video_id}_captions.txt"
    )
    
    progress(1)
    logger.debug(f"Video SENT TO VECTOR DB VECTORS SENT: {vectors}")

    os.environ["VIDEO_CURRENT_ID"] = str(video_id)

    return f"✅ Video URL Submitted: {url}"

# Chatbot function
def chatbot_response(message, history):
    vdb = VectorDB(
        db_name="mindflix-nomic",
        embed_model=EMBED_MODEL
    )
    response = vdb.query_db(
        input_text=message,
        video_id=os.getenv("VIDEO_CURRENT_ID"),
        top_k=3
    )
    return f"You expected: {response}" 

# Video URL submission interface
video_interface = gr.Interface(
    fn=process_video,
    inputs=gr.Textbox(label="Enter Video URL"),
    outputs=gr.Markdown(),
    live=False,
    allow_flagging="never"  # Avoid auto-processing
)

# Chatbot interface
chat_interface = gr.ChatInterface(
    title="Video-RAG",
    fn=chatbot_response,
    type="messages"
)

# Multi-tab Gradio App
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 Multi-Tab Video Chat App")

    with gr.Tabs():
        with gr.Tab("📤 Submit Video"):
            video_interface.render()  # Render Video URL Interface
        
        with gr.Tab("💬 Chatbot"):
            chat_interface.render()  # Render Chatbot Interface

# Run the app
demo.launch()
