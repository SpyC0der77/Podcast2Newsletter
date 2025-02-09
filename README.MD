# Podcast2Newsletter

Podcast2Newsletter is an automated tool that transforms podcast episodes into engaging newsletters. By fetching the latest episodes from RSS feeds, transcribing the audio using Deepgram, and summarizing the content with Google’s Gemini API, this project produces a formatted Markdown newsletter that can be emailed directly to subscribers.

---

## Features

- **Automated Podcast Processing:**\
  Retrieves podcast episodes from user-defined RSS feeds.

- **Audio Transcription:**\
  Uses Deepgram to transcribe podcast audio into text, preserving timing information.

- **Newsletter Generation:**\
  Leverages Google Gemini’s generative AI to create a structured newsletter including a title, summary, and detailed sections with timestamps.

- **Template-Based Markdown Output:**\
  Formats the generated content with a customizable Handlebars template, providing clickable timestamps for easy navigation.

- **Email Delivery:**\
  Sends the final newsletter via Gmail using the Gmail API, enabling seamless distribution.

- **GitHub Actions Integration:**\
  A pre-configured GitHub Actions workflow schedules daily newsletter generation and delivery.

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── main.yml         # GitHub Actions workflow for daily execution
├── .env.example             # Example environment configuration file
├── .gitignore               # Files and folders to exclude from version control
├── action.py                # Primary script used by GitHub Actions to process feeds, transcribe audio, and send newsletters
├── feeds.json               # JSON file listing podcast RSS feeds and recipient email addresses
├── geminifull.py            # Alternative script for processing an uploaded podcast audio file and generating a newsletter
├── LICENSE                  # MIT License
├── main.py                  # Standalone script for processing a single podcast episode into a newsletter
├── README.MD                # This README file
└── requirements.txt         # Python dependencies
```

---

## Requirements

- **Python:** 3.7 or higher
- **APIs and Services:**
  - [Google Gemini API](https://developers.generativeai.google/)
  - [Deepgram Transcription API](https://deepgram.com/)
  - [Gmail API](https://developers.google.com/gmail/api)
- **Python Libraries:**\
  The required libraries are listed in [requirements.txt](requirements.txt) and include:
  - `feedparser`
  - `pybars3`
  - `python-dotenv`
  - `deepgram-sdk`
  - `google-generativeai`
  - `google-api-python-client`
  - `google-auth-httplib2`
  - `google-auth-oauthlib`
  - `markdown`

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/podcast2newsletter.git
   cd podcast2newsletter
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

1. **Environment Variables:**\
   Create a `.env` file in the root directory (you can refer to [`.env.example`](.env.example)) and set the following variables:

   ```env
   GEMINI_API_KEY=<Your Google Gemini API Key>
   PODCAST_URL=<Your Podcast RSS Feed URL>
   DEEPGRAM_API_KEY=<Your Deepgram API Key>
   GMAIL_TOKEN=<Base64-encoded Gmail token pickle data>
   ```

2. **Podcast Feeds:**\
   Configure your podcast feeds and recipient emails in the `feeds.json` file. An example entry looks like:

   ```json
   [
     {
       "url": "https://feeds.megaphone.fm/STU4418364045",
       "email": "your-email@example.com"
     },
     {
       "url": "https://feeds.megaphone.fm/LMG3928170156",
       "email": "another-recipient@example.com"
     }
   ]
   ```

---

## Usage

### Local Execution

You can run the newsletter generation process locally using one of the following scripts:

- **`action.py`****:**\
  Designed for automated processing (e.g., via GitHub Actions). It processes all feeds defined in `feeds.json`, transcribes new podcast episodes (from the past 24 hours), generates a newsletter, and sends it via email.

  ```bash
  python action.py
  ```

- **`main.py`****:**\
  A standalone script that fetches a single podcast episode, processes it, and generates a Markdown newsletter.

  ```bash
  python main.py
  ```

- **`geminifull.py`****:**\
  An alternative script to process an uploaded podcast audio file. It reads episode metadata from `episode.json`, generates a newsletter, and saves it as `newsletter.md`.

  ```bash
  python geminifull.py
  ```

### GitHub Actions Integration

The repository includes a GitHub Actions workflow (`.github/workflows/main.yml`) that triggers the `action.py` script daily (using a cron schedule at midnight UTC) or via manual dispatch. This enables automated daily newsletter generation and email distribution.

---

## How It Works

1. **Fetching Podcast Episodes:**\
   The tool parses RSS feeds (configured in `feeds.json`) and identifies new episodes based on the publication date.

2. **Audio Transcription:**\
   For each new episode, Deepgram transcribes the audio, providing a transcript segmented by timestamps.

3. **Newsletter Generation:**\
   The transcript segments are sent to Google Gemini’s generative AI, which produces a detailed newsletter with:

   - A title and summary
   - Multiple sections, each with a descriptive header, detailed content, and clickable timestamps

4. **Markdown Templating:**\
   A Handlebars template compiles the generated content into a Markdown file (`newsletter.md`).

5. **Email Delivery:**\
   The newsletter is emailed to the specified recipient using the Gmail API.

---

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss your ideas.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- **Deepgram:** For providing high-quality transcription services.
- **Google Gemini API:** For the advanced generative AI capabilities.
- **GitHub Actions:** For enabling seamless CI/CD and automation.
- **The Open Source Community:** For continuous contributions and support.

---

