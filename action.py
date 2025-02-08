#!/usr/bin/env python3
import os
import json
import asyncio
import datetime
import calendar
import base64
import pickle
import feedparser
import markdown

from pybars import Compiler
from dotenv import load_dotenv
from deepgram import Deepgram

# Gemini / Generative AI imports
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

# Gmail API imports
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.discovery import build

# ---------------- Load Environment Variables ----------------
load_dotenv()

# ---------------- Gemini API Configuration ----------------
# Configure Gemini API with your API key.
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Set up the generation configuration for the Gemini model.
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
            "title": content.Schema(type=content.Type.STRING),
            "summary": content.Schema(type=content.Type.STRING),
            "sections": content.Schema(
                type=content.Type.ARRAY,
                items=content.Schema(
                    type=content.Type.OBJECT,
                    enum=[],
                    required=["timestamp", "header", "content"],
                    properties={
                        "timestamp": content.Schema(type=content.Type.NUMBER),
                        "header": content.Schema(type=content.Type.STRING),
                        "content": content.Schema(type=content.Type.STRING),
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

# ---------------- Gmail API Service (Token from Secret) ----------------
def get_gmail_service():
    """
    Load the Gmail API token from the GMAIL_TOKEN environment variable.
    The token is expected to be the base64-encoded data of a valid token.pickle.
    """
    token_b64 = os.environ.get("GMAIL_TOKEN")
    if not token_b64:
        raise Exception("GMAIL_TOKEN environment variable is not set!")
    try:
        token_data = base64.b64decode(token_b64)
        creds = pickle.loads(token_data)
    except Exception as e:
        raise Exception(f"Error decoding GMAIL_TOKEN: {e}")
    service = build('gmail', 'v1', credentials=creds)
    return service

# ---------------- Email Sending Function ----------------
def send_email(service, recipient_email, subject, markdown_body,
               sender_name="Podcast2Newsletter", sender_email="carter.stach@gmail.com"):
    """
    Convert the Markdown newsletter to HTML and send a multipart email.
    """
    html_body = markdown.markdown(markdown_body)
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f"{sender_name} <{sender_email}>"
    message["To"] = recipient_email

    part_plain = MIMEText(markdown_body, "plain")
    part_html = MIMEText(html_body, "html")
    message.attach(part_plain)
    message.attach(part_html)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    email_body = {'raw': raw_message}

    try:
        sent_message = service.users().messages().send(userId="me", body=email_body).execute()
        print(f"Email sent to {recipient_email}. Message Id: {sent_message['id']}")
    except Exception as error:
        print(f"An error occurred while sending email to {recipient_email}: {error}")

# ---------------- Helper: Format Timestamp ----------------
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

# ---------------- Process a Single Episode ----------------
async def process_episode(episode, recipient_email, gmail_service):
    if not episode.get("enclosures"):
        print(f"Episode '{episode.get('title', 'No Title')}' has no audio enclosure. Skipping.")
        return

    audio_url = episode["enclosures"][0].get("href")
    episode_title = episode.get("title", "Untitled Episode")
    episode_description = episode.get("description", "No description provided.")

    print(f"Transcribing episode: {episode_title}")

    deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
    source = {'url': audio_url}
    transcription_options = {"punctuate": True, "diarize": True, "paragraphs": True}
    try:
        response = await deepgram.transcription.prerecorded(source, transcription_options)
    except Exception as e:
        print(f"Deepgram transcription failed for '{episode_title}': {e}")
        return

    try:
        paragraphs = response['results']['channels'][0]['alternatives'][0]['paragraphs']['paragraphs']
    except KeyError:
        print(f"Unexpected transcription format for '{episode_title}'.")
        return

    transcript_segments = []
    for paragraph in paragraphs:
        start_time = paragraph.get('start', 0)
        sentences = paragraph.get('sentences', [])
        content_text = ' '.join(sentence.get('text', '') for sentence in sentences)
        transcript_segments.append({
            'timestamp': start_time,
            'content': content_text
        })

    system_instruction = (
        f"You are creating a newsletter for a podcast titled '{episode_title}'.\n"
        f"Description: {episode_description}\n"
        f"The transcript is divided into timed segments. For each segment:\n"
        f"1. Create a section with a descriptive header.\n"
        f"2. Write 1-2 detailed paragraphs explaining the content.\n"
        f"3. Use the provided timestamp.\n"
        f"4. Maintain a professional tone without advertisements.\n"
        f"Include an overall title and summary for the newsletter.\n"
        f"Do not include any sponsorships or advertisements."
    )

    print(f"Generating newsletter for episode: {episode_title}")
    chat_session = model.start_chat(history=[])
    gemini_input = f"System: {system_instruction}\nTranscript segments: {json.dumps(transcript_segments)}"
    gemini_response = chat_session.send_message(gemini_input)

    try:
        newsletter_data = json.loads(gemini_response.text)
    except Exception as e:
        print(f"Error parsing Gemini response for '{episode_title}': {e}")
        return

    if "sections" in newsletter_data:
        for section in newsletter_data["sections"]:
            section["formatted_timestamp"] = format_timestamp(section.get("timestamp", 0))
    else:
        print(f"Gemini response missing 'sections' for '{episode_title}'.")
        return

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

    try:
        output = template(newsletter_data)
    except Exception as e:
        print(f"Error generating Markdown for '{episode_title}': {e}")
        return

    email_subject = f"Newsletter for {episode_title}"
    send_email(gmail_service, recipient_email, email_subject, output)
    print(f"Processed and emailed newsletter for episode: {episode_title}")

# ---------------- Process a Feed ----------------
async def process_feed(feed_item, gmail_service):
    feed_url = feed_item.get("url")
    recipient_email = feed_item.get("email")
    if not feed_url or not recipient_email:
        print("Skipping feed item with missing 'url' or 'email'.")
        return

    print(f"Processing feed: {feed_url}")
    feed = feedparser.parse(feed_url)
    now = datetime.datetime.utcnow()
    new_episodes = []
    for entry in feed.entries:
        if "published_parsed" in entry:
            published = datetime.datetime.utcfromtimestamp(calendar.timegm(entry.published_parsed))
            if (now - published) <= datetime.timedelta(days=1):
                new_episodes.append(entry)

    if not new_episodes:
        print(f"No new episodes in the past 24 hours for feed: {feed_url}")
        return

    for episode in new_episodes:
        await process_episode(episode, recipient_email, gmail_service)

# ---------------- Main Function ----------------
async def main():
    try:
        with open("feeds.json", "r") as f:
            feeds_data = json.load(f)
    except Exception as e:
        print(f"Error loading feeds.json: {e}")
        return

    gmail_service = get_gmail_service()
    for feed_item in feeds_data:
        await process_feed(feed_item, gmail_service)

if __name__ == "__main__":
    asyncio.run(main())
