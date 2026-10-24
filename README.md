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

The question may contain multiple questions. Explicit questions, separate
lines, and clear coordinated question clauses are answered independently with
separate retrieval. Broad overview questions use transcript chunks sampled
across the video's timeline rather than only the most semantically similar
chunks.

This version uses the `youtube_transcripts_v2` collection to store chunk
timestamps and index hashes. Previously indexed videos will be indexed once
into this collection on their next question; the old collection is left intact.

`/health` provides a basic liveness check. The SQLite job queue is persistent
and shared by API workers when they use the same `RAG_JOB_DATABASE` path. Its
file must be on persistent storage in deployment. Queued jobs survive restarts;
stale processing jobs can be reclaimed after the lease expires.

`RAG_DATABASE_DIR` can point Chroma at a persistent directory or deployment
volume. Keep `.env`, API keys, job databases, and the Chroma database out of
source control.

## Production notes

SQLite avoids the process-local job registry and works for a small deployment
where all API workers share a persistent local volume. For multiple machines
or high job volume, replace it with a managed queue/store such as Redis and
dedicated workers. Configure `FRONTEND_ORIGINS`, `Client/config.js`, and
persistent volumes for the deployed environment. Add authentication and rate
limits before exposing the API publicly.

## Tests

Run unit and API tests with:

```powershell
python -m unittest discover -s tests -v
```

To evaluate retrieval against known timestamp evidence, copy
`tests/retrieval_eval.example.json` to `tests/retrieval_eval.json`, replace the
sample values with questions and timestamp labels from your indexed videos,
then run:

```powershell
python -m RAG_Service.evaluate_retrieval
```

Track hit rate and expected-timestamp recall before/after retrieval changes.
The example file is a template; evaluation scores are meaningful only after
you provide verified questions and expected evidence from your videos.
