"""
Daily Journal Telegram Bot
--------------------------
Voice note in → Whisper transcription → Claude follow-up questions → saved entry
Entries are saved locally as JSON and sent as a full formatted Telegram message.
"""

import os, json, asyncio, tempfile, logging
from datetime import date
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
import openai
import anthropic
import aiohttp

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
ANTHROPIC_KEY   = os.environ["ANTHROPIC_KEY"]
JOURNAL_API_URL = os.environ.get("JOURNAL_API_URL", "")

openai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

sessions: dict[int, dict] = {}
pending_deletes: dict[int, str] = {}

SYSTEM_PROMPT = """You are a thoughtful personal journal assistant. Your job is to help someone reflect on their day.

You will receive a transcript of their voice note about their day. Your task is to:
1. Acknowledge what they shared warmly and briefly (1-2 sentences max)
2. Identify which of these topics they have NOT covered or have only touched on briefly:
   - Their biggest learning or insight
   - What they would do differently
   - What went well / wins
   - How they are feeling (mood/energy)
   - Focus or intentions for tomorrow
3. Ask about the 1-2 MOST IMPORTANT missing topics — keep it conversational, like a friend asking. Never ask more than 2 questions at once.
4. When you have enough on all topics (or after 3 rounds of follow-up), end your message with exactly the token: [ENTRY_READY]

Keep your tone warm, brief, and conversational. This person is driving — quick, natural back-and-forth only."""

SUMMARY_PROMPT = """Based on this conversation, write a structured journal entry as JSON only (no markdown, no backticks).

Shape:
{
  "headline": "Short punchy title for the day (max 8 words)",
  "oneliner": "One sentence summary (max 20 words)",
  "summary": "2-3 paragraph narrative in first person, warm and personal",
  "learning": "Key learning or insight (1-3 sentences, or empty string)",
  "wins": "What went well (1-3 sentences, or empty string)",
  "differently": "What they would do differently (1-3 sentences, or empty string)",
  "tomorrow": "Focus for tomorrow (1-3 sentences, or empty string)",
  "feeling": "Mood and energy note (1 sentence, or empty string)"
}"""


def entry_path(user_id: int, entry_date: str) -> Path:
    return Path(f"journal_{entry_date}_{user_id}.json")

def todays_entry_path(user_id: int) -> Path:
    return entry_path(user_id, date.today().isoformat())

async def transcribe_voice(file_path: str) -> str:
    with open(file_path, "rb") as f:
        result = openai_client.audio.transcriptions.create(
            model="whisper-1", file=f, language="en"
        )
    return result.text

async def claude_respond(messages: list[dict]) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text

async def claude_summarise(messages: list[dict]) -> dict:
    flat = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": f"{SUMMARY_PROMPT}\n\nCONVERSATION:\n{flat}"}],
    )
    raw = response.content[0].text.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

async def save_entry(entry: dict, user_id: int) -> dict:
    entry["id"]      = str(int(asyncio.get_event_loop().time() * 1000))
    entry["date"]    = date.today().isoformat()
    entry["user_id"] = user_id

    # Save locally on Railway as JSON backup
    todays_entry_path(user_id).write_text(json.dumps(entry, indent=2))

    # POST to journal app if URL is set
    if JOURNAL_API_URL:
        try:
            async with aiohttp.ClientSession() as s:
                await s.post(JOURNAL_API_URL, json=entry)
        except Exception as e:
            logging.warning(f"Could not reach journal app: {e}")

    return entry

def format_full_entry(entry: dict) -> str:
    """Format a complete journal entry as a readable Telegram message."""
    lines = [
        f"📓 *{entry.get('headline', 'Journal Entry')}*",
        f"_{entry.get('oneliner', '')}_",
        f"\n📅 {entry.get('date', '')}",
        f"\n📝 *Summary*\n{entry.get('summary', '')}",
    ]
    if entry.get('learning'):
        lines.append(f"\n💡 *Biggest learning*\n{entry['learning']}")
    if entry.get('wins'):
        lines.append(f"\n🏆 *Wins & highlights*\n{entry['wins']}")
    if entry.get('differently'):
        lines.append(f"\n🔄 *What I'd do differently*\n{entry['differently']}")
    if entry.get('tomorrow'):
        lines.append(f"\n🎯 *Focus for tomorrow*\n{entry['tomorrow']}")
    if entry.get('feeling'):
        lines.append(f"\n😌 *Mindset & energy*\n{entry['feeling']}")
    lines.append("\n_Use /delete to remove this entry or /restart to record again._")
    return "\n".join(lines)


# ── Core flow ─────────────────────────────────────────────────────────────────

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text("Got your voice note — transcribing… 🎙")
    voice   = update.message.voice or update.message.audio
    tg_file = await ctx.bot.get_file(voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await tg_file.download_to_drive(tmp.name)
        transcript = await transcribe_voice(tmp.name)
    os.unlink(tmp.name)
    preview = transcript[:200] + ("…" if len(transcript) > 200 else "")
    await update.message.reply_text(f"Transcribed ✓\n\n_{preview}_", parse_mode="Markdown")
    messages = [{"role": "user", "content": f"Here's my day:\n\n{transcript}"}]
    sessions[user_id] = {"transcript": transcript, "messages": messages, "done": False}
    reply = await claude_respond(messages)
    sessions[user_id]["messages"].append({"role": "assistant", "content": reply})
    if "[ENTRY_READY]" in reply:
        reply = reply.replace("[ENTRY_READY]", "").strip()
        sessions[user_id]["done"] = True
    await update.message.reply_text(reply)
    if sessions[user_id]["done"]:
        await finalise_entry(update, user_id)

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text    = update.message.text.strip()
    if user_id not in sessions or sessions[user_id].get("done"):
        await update.message.reply_text("No active entry in progress. Send a voice note to start one.")
        return
    sessions[user_id]["messages"].append({"role": "user", "content": text})
    reply = await claude_respond(sessions[user_id]["messages"])
    sessions[user_id]["messages"].append({"role": "assistant", "content": reply})
    if "[ENTRY_READY]" in reply:
        reply = reply.replace("[ENTRY_READY]", "").strip()
        sessions[user_id]["done"] = True
    await update.message.reply_text(reply)
    if sessions[user_id]["done"]:
        await finalise_entry(update, user_id)

async def finalise_entry(update: Update, user_id: int):
    await update.message.reply_text("Writing up your journal entry… ✍️")
    try:
        entry = await claude_summarise(sessions[user_id]["messages"])
        saved = await save_entry(entry, user_id)
        # Send full formatted entry back in Telegram so it's readable on phone
        full_msg = format_full_entry(saved)
        await update.message.reply_text(full_msg, parse_mode="Markdown")
    except Exception as e:
        logging.error(f"Failed to save entry: {e}")
        await update.message.reply_text("Something went wrong saving the entry. Try /save to retry.")
    finally:
        sessions.pop(user_id, None)


# ── Commands ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hey! 👋 I'm your daily journal bot.\n\n"
        "Send me a voice note about your day — as long as you like — and I'll handle the rest.\n\n"
        "I'll transcribe it, ask a couple of follow-up questions if anything's missing, "
        "then send you back the full formatted entry right here in Telegram.\n\n"
        "Commands:\n"
        "/skip — save right now, skip remaining questions\n"
        "/restart — scrap draft and start fresh\n"
        "/delete — delete today's saved entry"
    )

async def cmd_skip(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in sessions and not sessions[user_id].get("done"):
        sessions[user_id]["done"] = True
        await finalise_entry(update, user_id)
    else:
        await update.message.reply_text("Nothing in progress. Send a voice note to start.")

async def cmd_save(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await cmd_skip(update, ctx)

async def cmd_restart(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    had = user_id in sessions
    sessions.pop(user_id, None)
    pending_deletes.pop(user_id, None)
    if had:
        await update.message.reply_text("Draft cleared. 🗑️\n\nSend a new voice note whenever you're ready.")
    else:
        await update.message.reply_text("Nothing to clear. Send a voice note whenever you're ready.")

async def cmd_delete(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id    = update.effective_user.id
    today      = date.today().isoformat()
    entry_file = todays_entry_path(user_id)
    sessions.pop(user_id, None)
    if not entry_file.exists():
        await update.message.reply_text(f"No saved entry for today ({today}).\nUse /restart to scrap a draft.")
        return
    try:
        headline = json.loads(entry_file.read_text()).get("headline", "today's entry")
    except Exception:
        headline = "today's entry"
    pending_deletes[user_id] = today
    await update.message.reply_text(
        f"⚠️ Delete *\"{headline}\"*?\n\nReply /confirm to delete or /cancel to keep it.",
        parse_mode="Markdown"
    )

async def cmd_confirm(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in pending_deletes:
        await update.message.reply_text("Nothing pending — use /delete first.")
        return
    target_date = pending_deletes.pop(user_id)
    f = entry_path(user_id, target_date)
    if f.exists():
        f.unlink()
        await update.message.reply_text(f"✅ Entry for {target_date} deleted.\n\nSend a voice note to start a new one.")
    else:
        await update.message.reply_text("Entry wasn't found — may already be removed.")

async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in pending_deletes:
        pending_deletes.pop(user_id)
        await update.message.reply_text("Cancelled — your entry is safe. 👍")
    else:
        await update.message.reply_text("Nothing to cancel.")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("skip",    cmd_skip))
    app.add_handler(CommandHandler("save",    cmd_save))
    app.add_handler(CommandHandler("restart", cmd_restart))
    app.add_handler(CommandHandler("delete",  cmd_delete))
    app.add_handler(CommandHandler("confirm", cmd_confirm))
    app.add_handler(CommandHandler("cancel",  cmd_cancel))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    print("Bot is running…")
    app.run_polling()
