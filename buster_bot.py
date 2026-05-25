"""
Buster - PlayerData AI Assistant
RAG Architecture v2 - Pinecone v5 compatible
"""

import os
import json
import time
import threading
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── HEALTH CHECK SERVER - starts immediately ──────────────────
PORT = int(os.environ.get("PORT", 10000))

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

health_thread = threading.Thread(target=run_health_server, daemon=True)
health_thread.start()
print(f"Health server started on port {PORT}")

# ── HEAVY IMPORTS ─────────────────────────────────────────────
print("DEBUG: Loading heavy imports...")
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
print("DEBUG: Slack imports done")
import gspread
from google.oauth2.service_account import Credentials
print("DEBUG: Google imports done")
from pinecone import Pinecone
print("DEBUG: Pinecone import done")
import anthropic
print("DEBUG: Anthropic import done")

# ── CREDENTIALS ───────────────────────────────────────────────
print("DEBUG: Loading credentials...")
SLACK_BOT_TOKEN     = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN     = os.environ["SLACK_APP_TOKEN"]
PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "buster-knowledge")
ANTHROPIC_API_KEY   = os.environ["ANTHROPIC_API_KEY"]
MEMORY_SHEET_ID     = os.environ["MEMORY_SHEET_ID"]
GOOGLE_CREDS_JSON   = os.environ["GOOGLE_CREDENTIALS_JSON"]
MAX_HISTORY         = 10
TOP_K_CHUNKS        = 8
NAMESPACE           = "buster-docs"
EMBEDDING_MODEL     = "multilingual-e5-large"
print("DEBUG: Credentials loaded")
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Buster, an AI assistant built specifically for PlayerData Account Executives. You communicate via Slack. Plain text only — no Markdown, no headers, no ### or ## or #, no --- dividers, no *** or ** bold, no * bullet points, no - bullet points at the start of lines. Use numbers for lists (1. 2. 3.) or write naturally in sentences. Use blank lines between paragraphs for readability.

Your job is to help AEs with questions about HubSpot, pricing, Gong, SOPs, internal processes, and PlayerData's physical performance context. You are knowledgeable, direct, and friendly — like a senior colleague who knows the playbook inside out.

When someone thanks you for your responses, always be courteous, and then respond with the company slogan, "Never a doubt!"

You will be given RELEVANT KNOWLEDGE from PlayerData's internal document library at the start of each message. Use this knowledge to answer questions accurately. If the knowledge provided does not contain enough information to answer confidently, say so honestly and suggest who to ask based on information you find in the aforementioned library of PlayerData documents.

HOW TO HANDLE VAGUE QUESTIONS:
When a question could have different answers depending on the situation, give a concise branched answer that covers the key scenarios in one response. Limit follow up questions unless you need more specific information in order to provide the AE with the appropriate knowledge. AEs need concise but correct information.

For territory questions, only answer based on the Sales Territories document. Never guess or infer territory ownership. If a state or sport is not clearly covered, say you are not certain and direct the AE to Ben Slingerland or Mat Young.

For process and SOP questions (upsells, trials, pricing), if you do not have complete step-by-step information in the provided knowledge, say so clearly and direct the AE to the relevant document or manager rather than filling in gaps.

Keep answers practical and concise. Numbered steps for processes. Plain conversational language throughout. Never ask more than one question at a time."""

# ── GOOGLE SHEETS MEMORY ──────────────────────────────────────
def get_gsheet():
    print("DEBUG: Connecting to Google Sheets...")
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
    print("DEBUG: Google Sheets connected")
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

# ── PINECONE RETRIEVAL ─────────────────────────────────────────
def get_relevant_context(query):
    try:
        print("DEBUG: Querying Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        # Embed the query using Pinecone inference
        query_embedding = pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"}
        )[0]["values"]

        # Query the index
        results = index.query(
            namespace=NAMESPACE,
            vector=query_embedding,
            top_k=TOP_K_CHUNKS,
            include_metadata=True
        )

        chunks = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            source = metadata.get("source", "Unknown")
            if text:
                chunks.append(f"[Source: {source}]\n{text}")

        if chunks:
            print(f"DEBUG: Pinecone returned {len(chunks)} chunks")
            return "RELEVANT KNOWLEDGE FROM PLAYERDATA DOCUMENTS:\n\n" + "\n\n---\n\n".join(chunks)
        else:
            print("DEBUG: Pinecone returned no chunks")
            return ""

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return ""

# ── CLAUDE API CALL ────────────────────────────────────────────
def ask_claude(user_message, conversation_history, relevant_context):
    try:
        print("DEBUG: Calling Claude API...")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        messages = []
        for msg in conversation_history[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})

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

        print("DEBUG: Claude response received")
        return response.content[0].text

    except Exception as e:
        print(f"Error calling Claude: {e}")
        return "Sorry, I ran into an issue. Please try again in a moment."

# ── SLACK RESPONSE ─────────────────────────────────────────────
def send_slack_dm(user_id, text):
    try:
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                "Content-Type": "application/json"
            },
            json={"channel": user_id, "text": text, "username": "Buster"}
        )
        result = response.json()
        if not result.get("ok"):
            print(f"Slack API error: {result.get('error')}")
    except Exception as e:
        print(f"Error sending Slack message: {e}")

# ── SLACK APP ──────────────────────────────────────────────────
print("DEBUG: Initialising Slack app...")
app = App(token=SLACK_BOT_TOKEN)
print("DEBUG: Slack app initialised")

@app.event("message")
def handle_dm(event, say, logger):
    if event.get("channel_type") != "im":
        return
    if event.get("bot_id"):
        return

    user_id = event.get("user")
    text    = event.get("text", "").strip()

    if not text or not user_id:
        return

    logger.info(f"DM from {user_id}: {text[:80]}")

    save_message(user_id, "user", text)
    history = get_history(user_id)
    relevant_context = get_relevant_context(text)
    logger.info(f"Retrieved {len(relevant_context)} chars of context")

    response_text = ask_claude(text, history, relevant_context)
    send_slack_dm(user_id, response_text)
    save_message(user_id, "assistant", response_text)

if __name__ == "__main__":
    print("DEBUG: Entering main block...")
    print(f"Buster RAG v2 starting - Pinecone index: {PINECONE_INDEX_NAME}")
    print("DEBUG: Connecting to Slack Socket Mode...")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    print("DEBUG: Socket Mode handler created, starting...")
    handler.start()
