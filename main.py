"""
Daily Journal Telegram Bot
--------------------------
Voice note in → Whisper transcription → STAR-method follow-up questions → saved entry
Entries are saved locally as JSON and sent as a full formatted Telegram message.

STAR Method: Situation | Task | Action | Result
The bot automatically detects STAR-worthy events and probes to complete each one.
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

SYSTEM_PROMPT = """You are a personal journal assistant helping someone capture their working day using the STAR method.

STAR stands for:
  S — SITUATION:  The context or background. What was the setting, challenge, or event?
  T — TASK:       What was the goal, responsibility, or problem to be solved?
  A — ACTION:     What did they actually do? Steps taken, decisions made, how they handled it.
  R — RESULT:     What was the outcome? What changed, improved, or was learned?

STAR EVENT DETECTION — this is your primary job:
Actively scan everything they say for STAR-worthy moments. These include:
  - Any meeting, call, or conversation that had stakes or a goal
  - A problem, challenge, or obstacle they faced
  - A decision they made or were part of
  - A task or project they worked on
  - An interaction with a client, colleague, manager, or supplier
  - Anything that went well or didn't go as planned
  - Anything they describe as significant, tricky, frustrating, or rewarding

The moment you spot a STAR-worthy event, immediately treat it as a STAR item and check which of the four elements are present or missing:
  - SITUATION: Do we know the context and background?
  - TASK: Do we know what they were trying to achieve?
  - ACTION: Do we know what specific steps or decisions they took?
  - RESULT: Do we know what actually happened as a result?

Your behaviour:
1. Acknowledge what they shared warmly in 1 sentence.
2. Identify all STAR-worthy events from what they said.
3. Work through them one at a time — complete all four STAR elements for one event before moving to the next.
4. For each missing element, ask a specific, natural question that references the actual event — never ask generically. E.g. "You mentioned the supplier call — what were you trying to get out of that?" not "What was your task today?"
5. Ask a maximum of 2 questions per reply. Never overwhelm.
6. When all detected STAR events are fully covered (or after 3 rounds of follow-up), end your reply with exactly this token on its own line: [ENTRY_READY]

Keep your tone warm, brief, and conversational — this person may be driving.
Never explain the STAR framework to them. Just ask the right questions naturally."""

SUMMARY_PROMPT = """Based on this conversation, write a structured STAR-method journal entry as JSON only (no markdown, no backticks).

The entry must follow the STAR structure: Situation → Task → Action → Result.
Write in first person, warm and personal, as if the person wrote it themselves.
If there were multiple STAR events in the day, weave them together naturally within each section.

Return this exact shape:
{
  "headline": "Short punchy title for the day (max 8 words)",
  "oneliner": "One sentence summary of the day's key theme (max 20 words)",
  "situation": "The context and background for the main event(s). What was happening, what was the setting or challenge? (2-4 sentences)",
  "task": "What the person was trying to achieve — their goal or responsibility in that situation. (2-3 sentences)",
  "action": "The specific steps, decisions, and actions they took. Concrete — what exactly did they do and how? (3-5 sentences)",
  "result": "What happened as a result. Outcomes, impact, feedback received. (2-3 sentences)",
  "learning": "The key insight or lesson from today they will carry forward. (1-3 sentences, or empty string)",
  "tomorrow": "What they are focusing on or doing differently tomorrow based on today. (1-2 sentences, or empty string)",
  "feeling": "Brief note on mood, energy, or mindset at end of day. (1 sentence, or empty string)"
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
        max_tokens=1200,
        messages=[{"role": "user", "content": f"{SUMMARY_PROMPT}\n\nCONVERSATION:\n{flat}"}],
    )
    raw = response.content[0].text.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

async def save_entry(entry: dict, user_id: int) -> dict:
    entry["id"]      = str(int(asyncio.get_event_loop().time() * 1000))
    entry["date"]    = date.today().isoformat()
    entry["user_id"] = user_id
    todays_entry_path(user_id).write_text(json.dumps(entry, indent=2))
    if JOURNAL_API_URL:
        try:
            async with aiohttp.ClientSession() as s:
                await s.post(JOURNAL_API_URL, json=entry)
        except Exception as e:
            logging.warning(f"Could not reach journal app: {e}")
    return entry

def format_full_entry(entry: dict) -> str:
    """Format a STAR-method journal entry as a readable Telegram message."""
    lines = [
        f"📓 *{entry.get('headline', 'Journal Entry')}*",
        f"_{entry.get('oneliner', '')}_",
        f"\n📅 {entry.get('date', '')}",
        f"\n━━━━━━━━━━━━━━━━━━━━",
        f"\n🔵 *SITUATION*\n{entry.get('situation', '')}",
        f"\n🎯 *TASK*\n{entry.get('task', '')}",
        f"\n⚡ *ACTION*\n{entry.get('action', '')}",
        f"\n✅ *RESULT*\n{entry.get('result', '')}",
        f"\n━━━━━━━━━━━━━━━━━━━━",
    ]
    if entry.get('learning'):
        lines.append(f"\n💡 *Key learning*\n{entry['learning']}")
    if entry.get('tomorrow'):
        lines.append(f"\n🔜 *Tomorrow's focus*\n{entry['tomorrow']}")
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

    # No active session — treat this text as the opening brain dump, start a new entry
    if user_id not in sessions or sessions[user_id].get("done"):
        await update.message.reply_text("Got it — reading through your day… 📝")
        messages = [{"role": "user", "content": f"Here's my day:\n\n{text}"}]
        sessions[user_id] = {"transcript": text, "messages": messages, "done": False}
        reply = await claude_respond(messages)
        sessions[user_id]["messages"].append({"role": "assistant", "content": reply})
        if "[ENTRY_READY]" in reply:
            reply = reply.replace("[ENTRY_READY]", "").strip()
            sessions[user_id]["done"] = True
        await update.message.reply_text(reply)
        if sessions[user_id]["done"]:
            await finalise_entry(update, user_id)
        return

    # Active session — continue the conversation
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
        "Send me a voice note or type/paste your day and I'll structure it using the *STAR method*:\n\n"
        "🔵 *Situation* — what was the context?\n"
        "🎯 *Task* — what were you trying to achieve?\n"
        "⚡ *Action* — what did you actually do?\n"
        "✅ *Result* — what happened?\n\n"
        "I'll automatically spot any key moments from your day and ask follow-up questions "
        "to make sure every event is fully captured.\n\n"
        "Commands:\n"
        "/skip — save right now, no more questions\n"
        "/restart — scrap draft and start fresh\n"
        "/delete — delete today's saved entry",
        parse_mode="Markdown"
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
        await update.message.reply_text(f"No saved entry for today ({today}).\nUse /restart to scrap a draft in progress.")
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
        await update.message.reply_text("Entry wasn't found — may have already been removed.")

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
