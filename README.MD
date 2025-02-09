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

- **`action.py`:**\
  Designed for automated processing (e.g., via GitHub Actions). It processes all feeds defined in `feeds.json`, transcribes new podcast episodes (from the past 24 hours), generates a newsletter, and sends it via email.

  ```bash
  python action.py
  ```

### Using the Hosted GitHub Actions Version

If you prefer to run Podcast2Newsletter automatically via GitHub Actions, follow these steps:

1. **Fork this repository** to your GitHub account.
2. **Enable GitHub Actions** in your repository settings.
3. **Configure Secrets:**
   - Go to `Settings > Secrets and variables > Actions` in your repository.
   - Add the following secrets:
     - `GEMINI_API_KEY`
     - `DEEPGRAM_API_KEY`
     - `GMAIL_TOKEN`
4. **Modify `feeds.json`** to include the podcasts you want to track and submit a pull request.
5. **Trigger the workflow manually**:
   - Navigate to the `Actions` tab in your GitHub repository.
   - Select the `Run Newsletter Generator` workflow and click `Run workflow`.
6. **Scheduled Execution:**
   - The workflow is configured to run daily at midnight UTC using GitHub Actions' cron scheduler.
   - You can modify the schedule in `.github/workflows/main.yml` if needed.

This setup ensures that your podcast newsletters are generated and sent automatically without requiring manual execution.

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

Happy Podcasting and Newslettering!

