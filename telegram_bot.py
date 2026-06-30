import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.error import TelegramError

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

_APPROVAL_FILE = Path(__file__).resolve().parent / "approval_status.json"
_TASKS_FILE = Path(__file__).resolve().parent / "tasks_data.json"
_bot_app: Application | None = None

# In-memory user state
_user_state: dict[int, dict] = {}

ALL_PLATFORMS = ["LinkedIn", "Facebook", "Instagram", "TikTok"]
PLATFORM_CALLBACK_PREFIX = "plt_"


# ── persistence helpers ─────────────────────────────────────


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _save_approval(task_id: str, status: str) -> None:
    data = _load_json(_APPROVAL_FILE)
    data[task_id] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(_APPROVAL_FILE, data)
    print(f"[telegram_bot] Saved approval: task_id={task_id} status={status}")


def _save_task_data(task_id: str, data: dict) -> None:
    tasks = _load_json(_TASKS_FILE)
    tasks[task_id] = data
    _save_json(_TASKS_FILE, tasks)
    print(f"[telegram_bot] Saved task data: task_id={task_id}")


def _load_task_data(task_id: str) -> dict | None:
    return _load_json(_TASKS_FILE).get(task_id)


def _build_approval_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve", callback_data="approve"),
            InlineKeyboardButton("❌ Reject", callback_data="reject"),
            InlineKeyboardButton("✏️ Edit", callback_data="edit"),
        ]
    ])


def _build_caption(platform: str, copy_text: str, task_id: str) -> str:
    caption = (
        f"📢 *Ново съдържание за одобрение*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"*Платформа:* {platform}\n"
        f"*Task ID:* `{task_id}`\n\n"
        f"*Копирайт:*\n{copy_text}"
    )
    if len(caption) > 1000:
        caption = caption[:997] + "..."
    return caption


def _build_platform_keyboard(selected: list[str]) -> InlineKeyboardMarkup:
    buttons = []
    for p in ALL_PLATFORMS:
        prefix = "✅" if p in selected else "⚪"
        buttons.append(
            InlineKeyboardButton(
                f"{prefix} {p}",
                callback_data=f"{PLATFORM_CALLBACK_PREFIX}{p}",
            )
        )
    # Grid 2 per row
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    # Select all / Generate row
    rows.append([
        InlineKeyboardButton(
            "✅ Всички 4" if len(selected) < 4 else "✅ Всички 4 ✓",
            callback_data="select_all",
        ),
    ])
    rows.append([
        InlineKeyboardButton(
            "▶️ Генерирай" if selected else "▶️ Избери платформа",
            callback_data="generate_now",
        ),
    ])
    return InlineKeyboardMarkup(rows)


# ── command: /start ─────────────────────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤖 *Flow Intelligence Content Bot*\n\n"
        "Използвай `/generate <тема>` за да създадеш съдържание.\n\n"
        "Пример: `/generate Как AI спестява време на счетоводни кантори`",
        parse_mode="Markdown",
    )


# ── command: /generate ──────────────────────────────────────


async def cmd_generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    # Parse topic
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text(
            "Моля, въведи тема. Пример:\n"
            "`/generate Как AI спестява време на счетоводни кантори`",
            parse_mode="Markdown",
        )
        return

    topic = parts[1].strip()
    _user_state[chat_id] = {
        "topic": topic,
        "selected_platforms": [],
        "awaiting_edit_for_task_id": None,
    }

    await update.message.reply_text(
        f"📌 *Тема:* {topic}\n\n"
        f"Избери платформа(и) за генериране:",
        reply_markup=_build_platform_keyboard([]),
        parse_mode="Markdown",
    )


# ── platform toggle callback ────────────────────────────────


async def handle_platform_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    data = query.data
    state = _user_state.get(chat_id)

    if not state:
        await query.edit_message_text("❗ Сесията е изтекла. Използвай /generate отново.")
        return

    # Toggle single platform
    if data.startswith(PLATFORM_CALLBACK_PREFIX):
        platform = data[len(PLATFORM_CALLBACK_PREFIX):]
        if platform in state["selected_platforms"]:
            state["selected_platforms"].remove(platform)
        else:
            state["selected_platforms"].append(platform)

    elif data == "select_all":
        if len(state["selected_platforms"]) == len(ALL_PLATFORMS):
            state["selected_platforms"] = []
        else:
            state["selected_platforms"] = list(ALL_PLATFORMS)

    elif data == "generate_now":
        if not state["selected_platforms"]:
            await query.edit_message_text(
                "⚠️ Избери поне една платформа преди да генерираш.",
                reply_markup=_build_platform_keyboard(state["selected_platforms"]),
            )
            return
        await _run_pipeline_for_selection(chat_id, state, query)
        return

    # Update the keyboard
    await query.edit_message_reply_markup(
        reply_markup=_build_platform_keyboard(state["selected_platforms"]),
    )


# ── pipeline runner ─────────────────────────────────────────


async def _run_pipeline_for_selection(
    chat_id: int,
    state: dict,
    query: Update.callback_query,
) -> None:
    topic = state["topic"]
    platforms = state["selected_platforms"]
    platform_list = ", ".join(platforms)

    await query.edit_message_text(
        f"⏳ Генерирам съдържание за: *{platform_list}*\n"
        f"Това може да отнеме няколко минути...",
        parse_mode="Markdown",
    )

    from main import run_pipeline_for_platform
    from telegram import Bot
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    for idx, platform in enumerate(platforms, 1):
        try:
            progress_text = (
                f"⏳ Генерирам ({idx}/{len(platforms)}) за *{platform}*..."
            )
            await bot.send_message(chat_id=chat_id, text=progress_text, parse_mode="Markdown")

            result = run_pipeline_for_platform(topic, platform)
            task_id = result["task_id"]
            copy_text = result["copy_text"]
            image_path = result["image_path"]

            # Save task data for edit flow
            _save_task_data(task_id, {
                "platform": platform,
                "copy_text": copy_text,
                "image_path": image_path,
            })

            caption = _build_caption(platform, copy_text, task_id)
            keyboard = _build_approval_keyboard()

            if image_path and Path(image_path).exists():
                with open(image_path, "rb") as f:
                    msg = await bot.send_photo(
                        chat_id=chat_id,
                        photo=f,
                        caption=caption,
                        reply_markup=keyboard,
                        parse_mode="Markdown",
                    )
            else:
                msg = await bot.send_message(
                    chat_id=chat_id,
                    text=caption,
                    reply_markup=keyboard,
                    parse_mode="Markdown",
                )

            print(f"[telegram_bot] Sent {platform} for approval, message_id={msg.message_id}")

        except Exception as e:
            await bot.send_message(
                chat_id=chat_id,
                text=f"❌ Грешка при генериране за *{platform}*: {e}",
                parse_mode="Markdown",
            )

    await bot.send_message(
        chat_id=chat_id,
        text="✅ *Готово!* Всички публикации са изпратени за одобрение.",
        parse_mode="Markdown",
    )

    # Clean up state
    _user_state.pop(chat_id, None)


# ── approval / edit callback ────────────────────────────────


async def handle_approval_response(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    choice = query.data

    # Extract task_id from caption
    raw = query.message.caption or query.message.text or ""
    task_id = "unknown"
    for line in raw.split("\n"):
        if line.startswith("*Task ID:*"):
            task_id = line.replace("*Task ID:*", "").strip().strip("`")
            break

    status_map = {
        "approve": "approved",
        "reject": "rejected",
        "edit": "edit_requested",
    }
    label_map = {
        "approve": "✅ Одобрено",
        "reject": "❌ Отхвърлено",
        "edit": "✏️ Редакция поискана",
    }
    edit_text_map = {
        "approve": "✅ Одобрено — съдържанието ще бъде публикувано.",
        "reject": "❌ Отхвърлено — съдържанието няма да бъде публикувано.",
        "edit": "✏️ Редакция поискана.\nИзпрати ми новия текст в чата.",
    }

    status = status_map.get(choice, "unknown")
    _save_approval(task_id, status)

    new_text = f"*{label_map.get(choice, choice)}*\n{edit_text_map.get(choice, '')}"

    # Update the message
    await query.edit_message_caption(
        caption=new_text,
        reply_markup=None,
        parse_mode="Markdown",
    )

    print(f"[telegram_bot] handle_approval_response: {choice} for task_id={task_id}")

    # If Edit, set state to await new text
    if choice == "edit":
        if chat_id not in _user_state:
            _user_state[chat_id] = {}
        _user_state[chat_id]["awaiting_edit_for_task_id"] = task_id


# ── text message handler (edit flow) ────────────────────────


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = _user_state.get(chat_id)

    if not state or not state.get("awaiting_edit_for_task_id"):
        return

    task_id = state["awaiting_edit_for_task_id"]
    new_copy = update.message.text.strip()

    if not new_copy:
        await update.message.reply_text("Моля, изпрати не-празен текст.")
        return

    # Update task data
    task_data = _load_task_data(task_id)
    if not task_data:
        await update.message.reply_text(f"❌ Task {task_id} не е намерен.")
        state["awaiting_edit_for_task_id"] = None
        return

    platform = task_data["platform"]
    image_path = task_data["image_path"]

    # Update copy
    task_data["copy_text"] = new_copy
    _save_task_data(task_id, task_data)

    # Clear edit state
    state["awaiting_edit_for_task_id"] = None

    # Send new approval message with updated copy
    from telegram import Bot
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    caption = _build_caption(platform, new_copy, task_id)
    keyboard = _build_approval_keyboard()

    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            msg = await bot.send_photo(
                chat_id=chat_id,
                photo=f,
                caption=caption,
                reply_markup=keyboard,
                parse_mode="Markdown",
            )
    else:
        msg = await bot.send_message(
            chat_id=chat_id,
            text=caption,
            reply_markup=keyboard,
            parse_mode="Markdown",
        )

    await update.message.reply_text(
        f"✅ Копирайтът за *{platform}* беше обновен! "
        f"Новият текст е изпратен за одобрение (message_id={msg.message_id}).",
        parse_mode="Markdown",
    )


# ── setup ───────────────────────────────────────────────────


async def _error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.error("Exception while handling an update: %s", context.error)

def setup_bot_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("generate", cmd_generate))
    application.add_handler(CallbackQueryHandler(handle_platform_callback, pattern=f"^{PLATFORM_CALLBACK_PREFIX}"))
    application.add_handler(CallbackQueryHandler(handle_platform_callback, pattern="^select_all$"))
    application.add_handler(CallbackQueryHandler(handle_platform_callback, pattern="^generate_now$"))
    application.add_handler(CallbackQueryHandler(handle_approval_response, pattern="^(approve|reject|edit)$"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_error_handler(_error_handler)
    print("[telegram_bot] All handlers registered")


def run_bot_polling() -> None:
    """Start the bot polling loop (blocking)."""
    global _bot_app

    if not TELEGRAM_BOT_TOKEN:
        print("[telegram_bot] No TELEGRAM_BOT_TOKEN set — cannot start polling.")
        return

    import time
    print("[telegram_bot] Waiting 5s before connecting to Telegram...")
    time.sleep(5)

    _bot_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    setup_bot_handlers(_bot_app)

    import logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.WARNING,
    )

    print("[telegram_bot] Starting polling (long-lived process)...")
    _bot_app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


def check_approval_status(task_id: str) -> str | None:
    data = _load_json(_APPROVAL_FILE)
    entry = data.get(task_id)
    return entry["status"] if entry else None


async def test_telegram_connection() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

    from telegram import Bot
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    text = "🤖 Flow Intelligence Content Bot - връзка установена!"
    msg = await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    print(f"[telegram_bot] Test message sent, message_id={msg.message_id}")
    print(f"[telegram_bot] ✅ Съобщението пристигна в Telegram!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_telegram_connection())
