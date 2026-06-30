import asyncio
import sys
import time
import uuid

from content_generator import generate_idea, generate_final_copy, classify_platform
from media_generator import generate_image, generate_image_prompt_from_copy
from telegram_bot import run_bot_polling


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def run_pipeline_for_platform(topic: str, platform: str) -> dict:
    """Execute the full marketing content pipeline for a specific platform.

    Args:
        topic: The topic/prompt for content generation.
        platform: Target social platform (linkedin, instagram, etc.).

    Returns:
        Dict with task_id, platform, copy_text, image_path.
    """
    task_id = uuid.uuid4().hex[:8]
    platform_lower = platform.lower()

    print("=" * 64)
    print(f"  🤖 FLOW INTELLIGENCE — CONTENT PIPELINE")
    print(f"  Task ID: {task_id}")
    print(f"  Topic:   {topic}")
    print(f"  Platform: {platform}")
    print("=" * 64)

    # ── Step 1: Generate idea ──────────────────────────────────
    print(f"\n[{_timestamp()}] [1/5] Генериране на маркетинг идея за {platform}...")
    idea = generate_idea(topic, platform_lower)
    print(f"  → Идея: {idea[:120]}...")

    # ── Step 2: Classify platform (for format hint only) ───────
    print(f"\n[{_timestamp()}] [2/5] Определяне на формат...")
    classification = classify_platform(idea)
    fmt = classification["format"]
    print(f"  → Платформа: {platform_lower}  |  Формат: {fmt}")

    # ── Step 3: Generate final copy ────────────────────────────
    print(f"\n[{_timestamp()}] [3/5] Генериране на финален копирайт за {platform}...")
    final_copy = generate_final_copy(idea, platform_lower)
    print(f"  → Копирайт ({len(final_copy)} символа):")
    for line in final_copy.split("\n"):
        print(f"    {line}")

    # ── Step 4: Build image prompt & generate ─────────────────
    print(f"\n[{_timestamp()}] [4/5] Изграждане на image prompt и генериране...")
    image_prompt = generate_image_prompt_from_copy(final_copy, platform_lower)
    print(f"  → Image prompt: {image_prompt[:100]}...")
    image_path = generate_image(image_prompt, save_local=True)
    print(f"  → Image path: {image_path}")

    # ── Step 5: Final result ────────────────────────────────────
    print(f"\n[{_timestamp()}] [5/5] Pipeline за {platform} завършен!")
    print("=" * 64)
    print(f"  ✅ Task ID:   {task_id}")
    print(f"  ✅ Platform:  {platform}")
    print(f"  ✅ Image:     {image_path}")
    print("=" * 64)

    return {
        "task_id": task_id,
        "platform": platform,
        "copy_text": final_copy,
        "image_path": image_path,
    }


if __name__ == "__main__":
    print("🚀 Стартиране на Flow Intelligence Content Bot...")
    run_bot_polling()
