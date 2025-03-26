import os
import json
import feedparser
import requests
import asyncio
from pybars import Compiler
from dotenv import load_dotenv
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from deepgram import Deepgram

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API using GEMINI_API_KEY from environment variables
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the generation configuration with a response schema using content.Schema
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type=content.Type.OBJECT,
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

# Use Gemini 2.0 flash-lite model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite",
    generation_config=generation_config,
)

async def main():
    # Fetch the latest podcast episode from the RSS feed
    rss = feedparser.parse('https://www.allearsenglish.com/feed/podcast')
    if not rss.entries:
        print("No episodes found in the RSS feed.")
        return

    episode = rss.entries[0]
    # Ensure the episode has an audio URL
    audio_url = episode.enclosures[0].href if episode.enclosures else None
    if not audio_url:
        print("No audio URL found in the latest episode.")
        return

    episode_title = episode.title
    episode_description = episode.description

    # Transcribe the podcast using Deepgram
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise EnvironmentError("DEEPGRAM_API_KEY not found in environment variables.")

    deepgram = Deepgram(DEEPGRAM_API_KEY)
    source = {'url': audio_url}
    transcription_options = {"punctuate": True, "diarize": True, "paragraphs": True}

    print("Transcribing podcast...")
    try:
        transcription_response = await deepgram.transcription.prerecorded(source, transcription_options)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return

    # Extract paragraphs with timestamps from the transcription response
    try:
        paragraphs = transcription_response['results']['channels'][0]['alternatives'][0]['paragraphs']['paragraphs']
    except KeyError:
        print("Error: Unexpected transcription response structure.")
        return

    transcript_segments = []
    for paragraph in paragraphs:
        start_time = paragraph['start']
        sentences = paragraph.get('sentences', [])
        content_text = ' '.join(sentence['text'] for sentence in sentences)
        transcript_segments.append({
            'timestamp': start_time,
            'content': content_text
        })

    # Prepare the system instruction for Gemini newsletter generation
    system_instruction = f"""
You are creating a newsletter for a podcast titled "{episode_title}".
Description: {episode_description}
The transcript is divided into timed segments. For each segment:
1. Create a section with a descriptive header.
2. Write 1-2 detailed paragraphs explaining the content.
3. Use the provided timestamp.
4. Maintain a professional tone without advertisements.
Include an overall title and summary for the newsletter.
Do not include any sponsorships or advertisements.
"""

    print("Generating newsletter...")
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(
        f"System: {system_instruction}\nTranscript segments: {json.dumps(transcript_segments)}"
    )

    try:
        newsletter_data = json.loads(response.text)
    except json.JSONDecodeError:
        print("Error decoding Gemini response as JSON.")
        return

    # Helper function to format seconds into HH:MM:SS
    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    # Format timestamps in each section
    for section in newsletter_data["sections"]:
        section["formatted_timestamp"] = format_timestamp(section["timestamp"])

    # Handlebars template for generating the newsletter in Markdown
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

    # Save the newsletter to a Markdown file
    newsletter_file = "newsletter.md"
    with open(newsletter_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Newsletter generated successfully and saved to {newsletter_file}!")

if __name__ == "__main__":
    asyncio.run(main())
