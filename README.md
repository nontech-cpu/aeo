# Hermes AEO/GEO Intelligence Agent (`AEO_Checker`)

AI Search Visibility & Competitive Audit Platform built with **FastAPI**, **Playwright**, and **Jinja2 Print-CSS PDF**.

## Key Features
- **Deep Website Crawl & Prompt Generation:** Renders target URLs, extracts business summary & key services, suggests competitors, and generates 10 unbiased buyer discovery prompts.
- **Live Multi-Tab Browser Audit (`ChatGPT` & `Google Gemini`):** Executes each prompt in a fresh isolated/persistent tab with real-time progress streaming.
- **Optional Google Search Visibility Checks (Off by Default):** Toggleable **Google AI Overview** (`SERPAPI_API_KEY`) and **Google Page 1 Organic Top 10** (`SERPER_API_KEY` / `SERPAPI_API_KEY`) tracking from Step 2.
- **Executive A4 PDF Report & JSON Export:** Generates structured A4 PDFs via Playwright Chromium (`report_pdf.html`) with scorecard KPIs, optional Google SERP matrix, competitor leaderboard, actionable fixes, full prompt responses, and a ready-to-paste ChatGPT Consultant prompt.

## Quick Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```
2. Copy `.env.example` to `.env` and add your API keys:
   - `OPENAI_API_KEY` / `GEMINI_API_KEY`
   - Optional: `SERPAPI_API_KEY` (Google AI Overview + Page 1) and/or `SERPER_API_KEY` (Google Page 1 Top 10)
3. Start the server:
   ```bash
   python run.py
   ```
4. Open **`http://localhost:8000`** in your browser.
