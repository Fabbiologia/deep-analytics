# Deep Analytics Chat Embed – Usage Guide

This guide lets a GUI developer embed the ecological analytics chat into any page (e.g., datamares.org) with a simple copy‑paste snippet.

## 1) Start the API

The API wraps `run_analysis()` from `main.py` using FastAPI in `embed_api.py`.

- Install deps (in a Python virtualenv, per project rules):
  ```bash
  python -m venv env && source env/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Start the API server:
  ```bash
  uvicorn embed_api:app --host 0.0.0.0 --port 8080 --reload
  ```
- Optional CORS origins:
  ```bash
  export ALLOW_ORIGINS="https://datamares.org,http://localhost:3000"
  ```

Health check:
```
GET http://localhost:8080/health
```

OpenAPI docs available at:
```
GET http://localhost:8080/docs
```

## 2) Host embed assets

`embed_api.py` mounts `/static` from the `embed/` directory:
- JS: `GET /static/embed.js`
- CSS: `GET /static/embed.css`

If the API runs at `https://YOUR_HOST`, the assets are available at:
- `https://YOUR_HOST/static/embed.js`
- `https://YOUR_HOST/static/embed.css`

## 3) Copy‑paste snippet for datamares.org

Add this to the HTML of the target page:
```html
<link rel="stylesheet" href="https://YOUR_HOST/static/embed.css">
<script src="https://YOUR_HOST/static/embed.js" defer
        data-endpoint="https://YOUR_HOST/chat"
        data-title="Gulf of California Data Assistant"
        data-primary="#0e7c86"
        data-position="bottom-right"></script>
```

- `data-endpoint` points to the API POST endpoint `/chat`.
- `data-title` sets the widget title.
- `data-primary` sets the brand color.
- `data-position` may be `bottom-right`, `bottom-left`, `top-right`, `top-left`.

## 4) API Contract

- Endpoint: `POST /chat`
- Request:
  ```json
  { "message": "Compare biomass trends for Cabo Pulmo vs Loreto since 2000" }
  ```
- Response:
  ```json
  { "answer": "...", "elapsed_ms": 1234 }
  ```
- Errors:
  - `400` if message is empty
  - `503` if the agent is unavailable
  - `500` for internal analysis errors

## 5) Notes

- The embed uses fetch with `Content-Type: application/json`.
- CORS must include the website origin (e.g., `https://datamares.org`).
- For production, put the API behind HTTPS.
- OpenAI limits: keep prompts concise; server already uses a compact system prompt.

## 6) Local quick test

Open `embed/example.html` in a browser after starting the server:
```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DA Embed Test</title>
    <link rel="stylesheet" href="http://localhost:8080/static/embed.css">
  </head>
  <body>
    <h1>Deep Analytics Chat – Local Test</h1>
    <script src="http://localhost:8080/static/embed.js" defer
            data-endpoint="http://localhost:8080/chat"
            data-title="Deep Analytics Assistant"
            data-primary="#0e7c86"
            data-position="bottom-right"></script>
  </body>
</html>
```

## 7) Streamlit UI (Local & Deploy)

You can run a simple UI for testing via Streamlit. It uses the tools defined in `tools.py`.

### Local run

Run inside a virtual environment (per project rules):

```bash
python -m venv env && source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: enable SQL tab
export DB_URI="postgresql+psycopg2://user:pass@host:5432/dbname"  # or DATABASE_URL

python -m streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Logs are saved to `streamlit_app.log`.

### Deploy on Hugging Face Spaces (free)

1. Create a new Space using the “Streamlit” template.
2. Ensure the repo root contains: `streamlit_app.py`, `tools.py`, `requirements.txt`, and any data/assets.
3. In Space → Settings → Variables & secrets, add as needed:
   - `OPENAI_API_KEY`
   - `DB_URI` or `DATABASE_URL`
4. The Space will auto-build. Share the Space URL.

### Deploy on Streamlit Community Cloud (free)

1. Push the repo to GitHub.
2. In https://streamlit.io/cloud, create a new app pointing to `streamlit_app.py`.
3. Add Secrets (in Settings → Secrets):
   - `OPENAI_API_KEY=...`
   - `DB_URI=...` (optional)
4. Deploy. The app will have a public URL.

Notes:
- All paths are relative; avoid hard‑coded absolute paths.
- Random seeds are set for reproducibility.
- For production, prefer HTTPS and set CORS appropriately on the API side.
