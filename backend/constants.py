EXTRACT_PROMPT="""  
Analyze this image thoroughly. Provide a concise, structured response (under 150 words) covering:  

1. **Objects/Entities**: List all items (counts, types, positions), including people, animals, objects, and notable features (e.g., clothing, size).  
2. **Scene**: Indoor/outdoor setting, time of day, weather, and lighting (e.g., sunny, dim).  
3. **Actions/Interactions**: Describe activities, gestures, expressions, and interactions between entities.  
4. **Visual Style**: Dominant colors, textures, patterns, and striking visual elements.  
5. **Mood**: Atmosphere (e.g., cheerful, tense) and emotions suggested by elements or expressions.  
6. **Key Highlights**: Unusual details, anomalies, or focal points. Suggest a brief plausible narrative.  

Keep descriptions factual, specific, and avoid assumptions.
"""
EMBED_MODEL="text-embedding-3-small"