EXTRACT_PROMPT="""  
Analyze the image thoroughly using the detected bounding boxes. Provide a concise, structured response (under 150 words) covering:
Objects/Entities: Identify all detected objects (type, count, and position). For detections, note the confidence scores and distribution across the image.
Scene: Determine whether the setting is indoor/outdoor, estimate time of day if possible, and describe lighting conditions.
Actions/Interactions: Explain the activities, gestures, or movements of detected entities. Highlight any key interactions among them.
Visual Composition: Describe dominant colors, textures, and notable patterns. Mention any symmetry, focal points, or unusual placements.
Mood & Atmosphere: Assess the image’s emotional tone based on expressions, posture, or activity intensity.
Key Highlights: Identify any anomalies or standout features. Provide a plausible brief narrative based on the arrangement and detected elements.
Keep descriptions precise, data-driven, and avoid assumptions beyond what the bounding boxes indicate.
"""
### Multi Lingual Embedding Model
EMBED_MODEL="nomic-ai/nomic-embed-text-v2-moe"
EMBED_TEMPLATE="""
TRANSCRIPT:
{}
SCENE DESCRIPTION:
{}
"""