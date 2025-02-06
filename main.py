import os
import json
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from pybars import Compiler
from dotenv import load_dotenv
from deepgram import Deepgram
import feedparser
import asyncio

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type=content.Type.OBJECT,
        enum=[],
        required=["title", "summary", "sections"],
        properties={
            "title": content.Schema(
                type=content.Type.STRING,
            ),
            "summary": content.Schema(
                type=content.Type.STRING,
            ),
            "sections": content.Schema(
                type=content.Type.ARRAY,
                items=content.Schema(
                    type=content.Type.OBJECT,
                    enum=[],
                    required=["timestamp", "header", "content"],
                    properties={
                        "timestamp": content.Schema(
                            type=content.Type.NUMBER,
                        ),
                        "header": content.Schema(
                            type=content.Type.STRING,
                        ),
                        "content": content.Schema(
                            type=content.Type.STRING,
                        ),
                    },
                ),
            ),
        },
    ),
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

async def main():
    # Fetch latest podcast episode from RSS
    rss = feedparser.parse('https://www.allearsenglish.com/feed/podcast')
    episode = rss.entries[0]
    audio_url = episode.enclosures[0].href
    episode_title = episode.title
    episode_description = episode.description

    # Transcribe podcast using Deepgram
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise EnvironmentError("DEEPGRAM_API_KEY not found in environment variables.")
    deepgram = Deepgram(DEEPGRAM_API_KEY)
    source = {'url': audio_url}
    transcription_options = {"punctuate": True, "diarize": True, "paragraphs": True}
    
    print("Transcribing podcast...")
    response = await deepgram.transcription.prerecorded(source, transcription_options)
    
    # Extract paragraphs with timestamps
    paragraphs = response['results']['channels'][0]['alternatives'][0]['paragraphs']['paragraphs']
    transcript_segments = []
    for paragraph in paragraphs:
        start_time = paragraph['start']
        sentences = paragraph['sentences']
        content_text = ' '.join(sentence['text'] for sentence in sentences)
        transcript_segments.append({
            'timestamp': start_time,
            'content': content_text
        })

    # Generate newsletter using Gemini
    system_instruction = f"""
    You are creating a newsletter for a podcast titled '{episode_title}'. 
    Description: {episode_description}
    The transcript is divided into timed segments. For each segment:
    1. Create a section with a descriptive header
    2. Write 1-2 detailed paragraphs explaining the content
    3. Use the provided timestamp
    4. Maintain professional tone without advertisements
    Include an overall title and summary for the newsletter.
    Don't include any sponsorships or advertisements.
    """

    print("Generating newsletter...")
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(
        f"System: {system_instruction}\nTranscript segments: {json.dumps(transcript_segments)}"
    )

    # Process Gemini response
    newsletter_data = json.loads(response.text)
    
    # Format timestamps
    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    
    for section in newsletter_data["sections"]:
        section["formatted_timestamp"] = format_timestamp(section["timestamp"])

    # Generate Markdown
    markdown_template = """
# {{title}}

{{summary}}

{{#each sections}}
## {{header}}

{{content}}

[Listen at {{formatted_timestamp}}]({{../base_url}}#t={{timestamp}})

{{/each}}
"""
    compiler = Compiler()
    template = compiler.compile(markdown_template)
    newsletter_data["base_url"] = audio_url
    
    output = template(newsletter_data)
    
    # Save newsletter
    with open("newsletter.md", "w", encoding="utf-8") as f:
        f.write(output)
    
    print("Newsletter generated successfully!")

if __name__ == "__main__":
    asyncio.run(main())
