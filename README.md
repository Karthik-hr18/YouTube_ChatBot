# YouTube Transcript Chatbot

Ask questions about a YouTube video's transcript and open cited timestamps.

## Local setup

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
4. Start the API from the project root:

   ```powershell
   python -m uvicorn Server.main:app --reload
   ```

5. Serve `Client/` with VS Code Live Server (or another static web server) and
   open `Client/index.html`. Do not use a `file://` URL.

The frontend API base URL is configured in `Client/config.js`. The backend
allows the origins in `FRONTEND_ORIGINS` (comma-separated); local defaults are
the two common Live Server origins.

## API

`POST /ask` accepts a YouTube URL or video ID:

```json
{
  "video_url": "https://www.youtube.com/watch?v=RE2c_eH0Sq4",
  "question": "What is this video about?",
  "refresh": false
}
```

The endpoint returns HTTP 202 with a `job_id` and `status_url`. Poll
`GET /ask/{job_id}` until the status is `completed` or `failed`. Set `refresh`
to `true` to fetch and replace the saved transcript; replacement chunks are
written before obsolete chunks are deleted.

This version uses the `youtube_transcripts_v2` collection to store chunk
timestamps and index hashes. Previously indexed videos will be indexed once
into this collection on their next question; the old collection is left intact.

`/health` provides a basic liveness check. `RAG_DATABASE_DIR` can point Chroma
at a persistent directory or deployment volume. Keep `.env`, API keys, and
the Chroma database out of source control.

## Production notes

The background-job registry is in process memory, suitable for local use and a
single-process demo. A multi-worker or restart-resilient deployment should
replace it with a shared job queue and status store (for example Redis-backed
workers). Configure `FRONTEND_ORIGINS`, `Client/config.js`, and a persistent
Chroma volume for the deployed environment. Add authentication and rate limits
before exposing the API publicly.

## Tests

Run unit and API tests with:

```powershell
python -m unittest discover -s tests -v
```
