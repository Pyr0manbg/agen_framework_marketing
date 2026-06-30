# Agent Framework — Marketing

AI-driven marketing content automation system.

## Architecture

The pipeline routes tasks to different AI providers based on task type:

| Task                  | Provider        |
|-----------------------|-----------------|
| Idea generation       | Big Pickle      |
| Platform classification | Big Pickle    |
| Final copy            | Claude          |
| Image generation      | Pollinations.ai |

A human-in-the-loop Telegram bot reviews content before publishing.

## Structure

```
agent_framework_marketing/
├── config.py              # Environment & constants
├── router.py              # Task-type → provider routing
├── content_generator.py   # Big Pickle & Claude clients
├── media_generator.py     # Image generation via Pollinations.ai
├── telegram_bot.py        # Telegram approval interface
├── main.py                # Orchestration pipeline
├── .env.example           # Required env vars template
├── requirements.txt       # Python dependencies
└── README.md
```

## Getting started

```bash
# 1. Clone / enter the project directory
cd agent_framework_marketing

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run dry‑run (no real API calls)
python main.py
```
