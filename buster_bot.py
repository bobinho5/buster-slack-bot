"""
Buster - PlayerData AI Assistant
RAG Architecture v1

Flow:
1. AE sends DM to Buster in Slack
2. Render script receives message via Socket Mode
3. Fetches conversation history from Google Sheets
4. Queries Pinecone for relevant knowledge chunks
5. Calls Claude API directly with context + history + question
6. Posts response back to AE via Slack API
7. Saves exchange to Google Sheets memory
"""

import os
import json
import time
import threading
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import gspread
from google.oauth2.service_account import Credentials
from pinecone import Pinecone
import anthropic

# ── CREDENTIALS ──────────────────────────────────────────────
SLACK_BOT_TOKEN      = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN      = os.environ["SLACK_APP_TOKEN"]
PINECONE_API_KEY     = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME  = os.environ.get("PINECONE_INDEX_NAME", "buster-knowledge")
ANTHROPIC_API_KEY    = os.environ["ANTHROPIC_API_KEY"]
MEMORY_SHEET_ID      = os.environ["MEMORY_SHEET_ID"]
GOOGLE_CREDS_JSON    = os.environ["GOOGLE_CREDENTIALS_JSON"]
PORT                 = int(os.environ.get("PORT", 8080))
MAX_HISTORY          = 10   # messages to pass to Claude
TOP_K_CHUNKS         = 8    # number of knowledge chunks to retrieve from Pinecone
# ─────────────────────────────────────────────────────────────

# ── SYSTEM PROMPT ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Buster, an AI assistant built specifically for PlayerData Account Executives. You communicate via Slack. Plain text only — no Markdown, no headers, no ### or ## or #, no --- dividers, no *** or ** bold, no * bullet points, no - bullet points at the start of lines. Use numbers for lists (1. 2. 3.) or write naturally in sentences. Use blank lines between paragraphs for readability.

Your job is to help AEs with questions about HubSpot, pricing, Gong, SOPs, internal processes, and PlayerData's physical performance context. You are knowledgeable, direct, and friendly — like a senior colleague who knows the playbook inside out.

You will be given RELEVANT KNOWLEDGE from PlayerData's internal document library at the start of each message. Use this knowledge to answer questions accurately. If the knowledge provided does not contain enough information to answer confidently, say so honestly and suggest who to ask.

HOW TO HANDLE VAGUE QUESTIONS:
When a question could have different answers depending on the situation, give a concise branched answer that covers the key scenarios in one response. Do not ask follow-up questions and wait for answers — AEs need immediate, usable answers.

For example:
If someone asks "how do I do an upsell?" — give a branched answer covering all scenarios (more than 12 months, 6-12 months, less than 6 months, annual vs monthly) with the key steps for each.
If someone asks "what is the price for an Air Unit?" — list pricing for all regions and variants concisely.

EXCEPTION: If a question is genuinely too open-ended to answer without context — for example "can you help me?" — ask one single specific question to narrow it down.

Keep answers practical and concise. Numbered steps for processes. Plain conversational language throughout. Never ask more than one question at a time."""
# ─────────────────────────────────────────────────────────────

# ── GOOGLE SHEETS MEMORY ─────────────────────────────────────
def get_gsheet():
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(MEMORY_SHEET_ID).sheet1
    try:
        headers = sheet.row_values(1)
        if not headers or headers[0] != "timestamp":
            sheet.insert_row(["timestamp", "user_id", "role", "message"], index=1)
    except Exception:
        sheet.insert_row(["timestamp", "user_id", "role", "message"], index=1)
    return sheet

def get_history(user_id):
    try:
        sheet = get_gsheet()
        all_rows = sheet.get_all_records()
        user_rows = [r for r in all_rows if str(r.get("user_id")) == str(user_id)]
        recent = user_rows[-MAX_HISTORY:]
        history = []
        for row in recent:
            role = row.get("role", "user")
            message = row.get("message", "")
            if role in ("user", "assistant") and message:
                history.append({"role": role, "content": str(message)})
        return history
    except Exception as e:
        print(f"Error reading history: {e}")
        return []

def save_message(user_id, role, message):
    try:
        sheet = get_gsheet()
        sheet.append_row([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            user_id,
            role,
            message
        ])
    except Exception as e:
        print(f"Error saving message: {e}")

# ── PINECONE RETRIEVAL ────────────────────────────────────────
def get_relevant_context(query):
    """Query Pinecone for the most relevant knowledge chunks."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        # Use Pinecone's integrated inference for embedding + search
        results = index.search(
            namespace="buster-docs",
            query={
                "inputs": {"text": query},
                "top_k": TOP_K_CHUNKS
            },
            fields=["text", "source", "section"]
        )

        chunks = []
        for hit in results.get("result", {}).get("hits", []):
            fields = hit.get("fields", {})
            text = fields.get("text", "")
            source = fields.get("source", "Unknown")
            section = fields.get("section", "")
            if text:
                chunks.append(f"[Source: {source} - {section}]\n{text}")

        if chunks:
            return "RELEVANT KNOWLEDGE FROM PLAYERDATA DOCUMENTS:\n\n" + "\n\n---\n\n".join(chunks)
        else:
            return ""

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return ""

# ── CLAUDE API CALL ───────────────────────────────────────────
def ask_claude(user_message, conversation_history, relevant_context):
    """Call Claude API directly with RAG context and conversation history."""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Build the message list
        messages = []

        # Add conversation history (excluding the current message)
        for msg in conversation_history[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Build the current user message with context prepended
        if relevant_context:
            full_user_message = f"{relevant_context}\n\nAE QUESTION:\n{user_message}"
        else:
            full_user_message = user_message

        messages.append({"role": "user", "content": full_user_message})

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages
        )

        return response.content[0].text

    except Exception as e:
        print(f"Error calling Claude: {e}")
        return "Sorry, I ran into an issue processing your question. Please try again in a moment."

# ── SLACK RESPONSE ────────────────────────────────────────────
def send_slack_dm(user_id, text):
    """Send a DM directly via Slack API using bot token."""
    try:
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "channel": user_id,
                "text": text,
                "username": "Buster",
            }
        )
        result = response.json()
        if not result.get("ok"):
            print(f"Slack API error: {result.get('error')}")
    except Exception as e:
        print(f"Error sending Slack message: {e}")

# ── SLACK APP ─────────────────────────────────────────────────
app = App(token=SLACK_BOT_TOKEN)

@app.event("message")
def handle_dm(event, say, logger):
    # Only respond to direct messages, ignore bot messages
    if event.get("channel_type") != "im":
        return
    if event.get("bot_id"):
        return

    user_id = event.get("user")
    text    = event.get("text", "").strip()

    if not text or not user_id:
        return

    logger.info(f"DM from {user_id}: {text[:80]}")

    # 1. Save the incoming user message to memory
    save_message(user_id, "user", text)

    # 2. Get conversation history
    history = get_history(user_id)

    # 3. Query Pinecone for relevant knowledge
    relevant_context = get_relevant_context(text)
    logger.info(f"Retrieved {len(relevant_context)} chars of context from Pinecone")

    # 4. Call Claude with context + history
    response_text = ask_claude(text, history, relevant_context)

    # 5. Send response to user via Slack
    send_slack_dm(user_id, response_text)

    # 6. Save Buster's response to memory
    save_message(user_id, "assistant", response_text)

# ── HEALTH CHECK SERVER ───────────────────────────────────────
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Buster RAG is running!")
    def log_message(self, format, *args):
        pass

def run_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    server.serve_forever()

if __name__ == "__main__":
    t = threading.Thread(target=run_health_server, daemon=True)
    t.start()
    print(f"Buster RAG starting - health server on port {PORT}")
    print(f"Pinecone index: {PINECONE_INDEX_NAME}")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
