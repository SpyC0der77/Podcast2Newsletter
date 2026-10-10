# Podcast2Newsletter

Generate Markdown newsletters from podcast episodes using Deepgram transcription and Google Gemini. The automated script processes recent episodes from RSS feeds and sends the resulting newsletters through Gmail.

## Setup

Use Python 3.12, matching the GitHub Actions workflow.

```bash
git clone https://github.com/SpyC0der77/Podcast2Newsletter.git
cd Podcast2Newsletter
python -m pip install -r requirements.txt
```

Copy [`.env.example`](.env.example) to `.env` and set `GEMINI_API_KEY` and `DEEPGRAM_API_KEY` to your API keys. The scripts use older Gemini and Deepgram SDK interfaces; dependency or service changes may require updates before they run successfully.

## Scripts

| Script | Behavior |
| --- | --- |
| `main.py` | Process the latest episode from its hardcoded RSS feed and write `newsletter.md` |
| `multi.py` | Process episodes from its hardcoded feed, write episode newsletters, and merge them into `merged_newsletter.md` |
| `action.py` | Process episodes published in the past 24 hours from `feeds.json` and email newsletters to the configured recipients |

To generate a local newsletter, edit the RSS URL in `main.py` and run:

```bash
python main.py
```

The scripts call external APIs and can incur usage charges. Audio URLs and timestamp links come from the podcast feed.

## Automated email delivery

Set `feeds.json` to the feeds and recipients you want to process:

```json
[
  {"url": "https://example.com/podcast.xml", "email": "reader@example.com"}
]
```

`action.py` also requires `GMAIL_TOKEN`, a base64-encoded pickle of authorized Gmail OAuth credentials with permission to send mail. Token creation is not included in this repo. Treat this token as a secret and only use a pickle you created and trust.

Run `python action.py` only when you intend to send email to the recipients in `feeds.json`.

The [GitHub Actions workflow](.github/workflows/main.yml) runs at 00:00 UTC daily and supports manual dispatch. Configure `GEMINI_API_KEY`, `DEEPGRAM_API_KEY`, and `GMAIL_TOKEN` as repository Actions secrets before enabling it. The workflow also passes `CREDENTIALS`, but `action.py` does not read that variable. `PODCAST_URL` is not used by the current scripts.

## License

[MIT](LICENSE).
