import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
OPENCODE_API_KEY: str = os.getenv("OPENCODE_API_KEY", "")
AYRSHARE_API_KEY: str = os.getenv("AYRSHARE_API_KEY", "")

OPENCODE_BASE_URL: str = "https://opencode.ai/zen/v1"
OPENCODE_MODEL: str = "big-pickle"
OPENCODE_ZEN_MODEL: str = "deepseek-v4-flash"
CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
POLLINATIONS_BASE_URL: str = "https://image.pollinations.ai/prompt"
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_IMAGE_MODEL: str = "gemini-2.5-flash-image"
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
