import gradio as gr
import time

# Function to handle video processing
def process_video(url, progress=gr.Progress()):
    if not url.strip():
        return "⚠️ Please enter a valid video URL."
    
    # Simulate loading with a progress bar
    for i in progress.tqdm(range(5), desc="Video Chunking"):
        time.sleep(0.5)
    
    print("Video URL SUBMITTED")

    return f"✅ Video URL Submitted: {url}"

# Chatbot function
def chatbot_response(message, history):
    return "You said: " + message

# Video URL submission interface
video_interface = gr.Interface(
    fn=process_video,
    inputs=gr.Textbox(label="Enter Video URL"),
    outputs=gr.Markdown(),
    live=False,
    allow_flagging="never"  # Avoid auto-processing
)

# Chatbot interface
chat_interface = gr.ChatInterface(fn=chatbot_response)

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
