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
from email.mime.multipart import MIMEMultipart  # Correct import for multipart emails
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
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

# Create the Gemini generative model.
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# ---------------- Gmail API Functions ----------------
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    """
    Authenticate with the Gmail API using credentials from environment variables.
    It first checks for a stored token in 'token.pickle'. If not found, it checks for
    a GMAIL_TOKEN environment variable (expected to be a base64-encoded token pickle).
    In a CI environment, if no valid token is available, an error is raised.
    """
    creds = None

    # Check if token.pickle exists
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    else:
        # If token.pickle does not exist, check if GMAIL_TOKEN is provided.
        if "GMAIL_TOKEN" in os.environ:
            try:
                token_data = base64.b64decode(os.environ["GMAIL_TOKEN"])
                with open('token.pickle', 'wb') as token_file:
                    token_file.write(token_data)
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                raise Exception(f"Error decoding GMAIL_TOKEN: {e}")

    # If no credentials or invalid credentials, check if we can do interactive login.
    if not creds or not creds.valid:
        # If running in CI, interactive login is not possible.
        if os.environ.get("CI"):
            raise Exception("Gmail credentials are not valid and interactive login is not possible in CI. Please update the GMAIL_TOKEN secret.")
        # Otherwise, refresh if possible or run the local server flow.
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            try:
                client_config = json.loads(os.environ["CREDENTIALS"])
            except KeyError:
                raise Exception("Environment variable CREDENTIALS not set!")
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the new credentials to token.pickle.
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    return service

def send_email(service, recipient_email, subject, markdown_body,
               sender_name="Podcast2Newsletter", sender_email="carter.stach@gmail.com"):
    """
    Create and send an email via the Gmail API.
    Converts the provided Markdown body to HTML and sends a multipart message.
    """
    # Convert the Markdown newsletter into HTML.
    html_body = markdown.markdown(markdown_body)

    # Create a multipart/alternative container.
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f"{sender_name} <{sender_email}>"
    message["To"] = recipient_email

    # Create the plain-text and HTML parts.
    part_plain = MIMEText(markdown_body, "plain")
    part_html = MIMEText(html_body, "html")

    message.attach(part_plain)
    message.attach(part_html)

    # Encode the message for Gmail.
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    email_body = {'raw': raw_message}

    try:
        sent_message = service.users().messages().send(userId="me", body=email_body).execute()
        print(f"Email sent to {recipient_email}. Message Id: {sent_message['id']}")
    except Exception as error:
        print(f"An error occurred while sending email to {recipient_email}: {error}")

# ---------------- Helper Functions ----------------
def format_timestamp(seconds):
    """
    Format a timestamp (in seconds) as HH:MM:SS.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

# ---------------- Main Processing Functions ----------------
async def process_episode(episode, recipient_email, gmail_service):
    """
    For a given episode, transcribe the audio, generate a newsletter using Gemini,
    convert the newsletter Markdown to HTML, and email it to the recipient.
    """
    if not episode.get("enclosures"):
        print(f"Episode '{episode.get('title', 'No Title')}' has no audio enclosure. Skipping.")
        return

    audio_url = episode["enclosures"][0].get("href")
    episode_title = episode.get("title", "Untitled Episode")
    episode_description = episode.get("description", "No description provided.")

    print(f"Transcribing episode: {episode_title}")

    # Initialize Deepgram and transcribe the audio.
    deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
    source = {'url': audio_url}
    transcription_options = {"punctuate": True, "diarize": True, "paragraphs": True}
    try:
        response = await deepgram.transcription.prerecorded(source, transcription_options)
    except Exception as e:
        print(f"Deepgram transcription failed for '{episode_title}': {e}")
        return

    # Extract transcript segments.
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

    # Build system instruction for Gemini.
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
    # Start Gemini chat session.
    chat_session = model.start_chat(history=[])
    gemini_input = f"System: {system_instruction}\nTranscript segments: {json.dumps(transcript_segments)}"
    gemini_response = chat_session.send_message(gemini_input)

    try:
        newsletter_data = json.loads(gemini_response.text)
    except Exception as e:
        print(f"Error parsing Gemini response for '{episode_title}': {e}")
        return

    # Add formatted timestamp to each section.
    if "sections" in newsletter_data:
        for section in newsletter_data["sections"]:
            section["formatted_timestamp"] = format_timestamp(section.get("timestamp", 0))
    else:
        print(f"Gemini response missing 'sections' for '{episode_title}'.")
        return

    # Generate Markdown using a Handlebars template.
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
    # Pass the audio URL as base_url.
    newsletter_data["base_url"] = audio_url

    try:
        output = template(newsletter_data)
    except Exception as e:
        print(f"Error generating Markdown for '{episode_title}': {e}")
        return

    # Prepare email subject and body.
    email_subject = f"Newsletter for {episode_title}"
    email_body = output  # Markdown output

    # Send the email.
    send_email(gmail_service, recipient_email, email_subject, email_body)
    print(f"Processed and emailed newsletter for episode: {episode_title}")

async def process_feed(feed_item, gmail_service):
    """
    Process a feed (with 'url' and 'email'):
    Parse the RSS feed, filter episodes from the past 24 hours, and process each.
    """
    feed_url = feed_item.get("url")
    recipient_email = feed_item.get("email")
    if not feed_url or not recipient_email:
        print("Skipping feed item with missing 'url' or 'email'.")
        return

    print(f"Processing feed: {feed_url}")
    feed = feedparser.parse(feed_url)
    now = datetime.datetime.utcnow()

    # Filter episodes published in the last 24 hours.
    new_episodes = []
    for entry in feed.entries:
        if "published_parsed" in entry:
            published = datetime.datetime.utcfromtimestamp(calendar.timegm(entry.published_parsed))
            if (now - published) <= datetime.timedelta(days=1):
                new_episodes.append(entry)

    if not new_episodes:
        print(f"No new episodes in the past 24 hours for feed: {feed_url}")
        return

    # Process each episode.
    for episode in new_episodes:
        await process_episode(episode, recipient_email, gmail_service)

async def main():
    # Load feeds from feeds.json.
    try:
        with open("feeds.json", "r") as f:
            feeds_data = json.load(f)
    except Exception as e:
        print(f"Error loading feeds.json: {e}")
        return

    # Get the Gmail service.
    gmail_service = get_gmail_service()

    # Process each feed.
    for feed_item in feeds_data:
        await process_feed(feed_item, gmail_service)

if __name__ == "__main__":
    asyncio.run(main())
