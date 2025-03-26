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
import glob

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

# Compile the Handlebars markdown template once to use for every episode
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

# Helper function to format seconds into HH:MM:SS
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# Function to create a safe filename from the episode title
def safe_filename(title):
    return "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()

async def main():
    # Parse the RSS feed (Megaphone feed)
    rss = feedparser.parse('https://feeds.megaphone.fm/allearsenglish')
    if not rss.entries:
        print("No episodes found in the RSS feed.")
        return

    # Instantiate Deepgram client once
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise EnvironmentError("DEEPGRAM_API_KEY not found in environment variables.")
    deepgram = Deepgram(DEEPGRAM_API_KEY)

    # Process the latest 10 episodes
    for idx, episode in enumerate(rss.entries[:10]):
        try:
            if not episode.enclosures or len(episode.enclosures) == 0:
                print(f"Episode '{episode.title}' does not have an audio URL. Skipping.")
                continue

            audio_url = episode.enclosures[0].href
            episode_title = episode.title
            episode_description = episode.description
            safe_name = safe_filename(episode_title)

            print(f"\nProcessing episode {idx+1}: {episode_title}")

            # Transcribe the podcast using Deepgram
            source = {'url': audio_url}
            transcription_options = {"punctuate": True, "diarize": True, "paragraphs": True}
            print("  Transcribing podcast...")
            transcription_response = await deepgram.transcription.prerecorded(source, transcription_options)

            # Extract paragraphs with timestamps from the transcription response
            try:
                paragraphs = transcription_response['results']['channels'][0]['alternatives'][0]['paragraphs']['paragraphs']
            except KeyError:
                print("  Error: Unexpected transcription response structure. Skipping this episode.")
                continue

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

            print("  Generating newsletter with Gemini...")
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(
                f"System: {system_instruction}\nTranscript segments: {json.dumps(transcript_segments)}"
            )

            try:
                newsletter_data = json.loads(response.text)
            except json.JSONDecodeError:
                print("  Error decoding Gemini response as JSON. Skipping this episode.")
                continue

            # Format timestamps in each section
            for section in newsletter_data["sections"]:
                section["formatted_timestamp"] = format_timestamp(section["timestamp"])

            # Add base_url for timestamp links
            newsletter_data["base_url"] = audio_url

            # Generate Markdown using the compiled template
            output = template(newsletter_data)

            # Save the newsletter to a Markdown file with a unique name
            newsletter_file = f"newsletter_{safe_name}.md"
            with open(newsletter_file, "w", encoding="utf-8") as f:
                f.write(output)

            print(f"  Newsletter generated and saved to {newsletter_file}!")
        except Exception as e:
            print(f"An error occurred while processing episode '{episode.title}': {e}")

def merge_newsletter_files():
    # Get all files starting with 'newsletter_'
    newsletter_files = glob.glob('newsletter_*')
    
    # Create output content
    merged_content = ''
    
    # Read and merge files
    for file in sorted(newsletter_files):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                merged_content += content + '\n\n'
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    # Write merged content to output file
    if merged_content:
        with open('merged_newsletter.md', 'w', encoding='utf-8') as f:
            f.write(merged_content.strip())
        print("Files merged successfully into 'merged_newsletter.md'")
    else:
        print("No newsletter files found to merge")

if __name__ == "__main__":
    asyncio.run(main())
    merge_newsletter_files()
