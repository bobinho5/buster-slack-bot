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

# ── CREDENTIALS ──────────────────────────────────────────────
SLACK_BOT_TOKEN   = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN   = os.environ["SLACK_APP_TOKEN"]
ZAPIER_WEBHOOK    = os.environ["ZAPIER_WEBHOOK_URL"]
MEMORY_SHEET_ID   = os.environ["MEMORY_SHEET_ID"]
GOOGLE_CREDS_JSON = os.environ["GOOGLE_CREDENTIALS_JSON"]
PORT              = int(os.environ.get("PORT", 8080))
MAX_HISTORY       = 10  # Number of recent messages to pass to Claude
# ─────────────────────────────────────────────────────────────

# ── GOOGLE SHEETS SETUP ──────────────────────────────────────
def get_sheet():
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

    # Ensure headers exist
    try:
        headers = sheet.row_values(1)
        if not headers or headers[0] != "timestamp":
            sheet.insert_row(
                ["timestamp", "user_id", "role", "message"],
                index=1
            )
    except Exception:
        sheet.insert_row(
            ["timestamp", "user_id", "role", "message"],
            index=1
        )
    return sheet

def get_history(user_id):
    """Retrieve the last MAX_HISTORY messages for a specific user."""
    try:
        sheet = get_sheet()
        all_rows = sheet.get_all_records()
        user_rows = [r for r in all_rows if str(r.get("user_id")) == str(user_id)]
        recent = user_rows[-MAX_HISTORY:]
        history = []
        for row in recent:
            role = row.get("role", "user")
            message = row.get("message", "")
            if role in ("user", "assistant") and message:
                history.append({"role": role, "content": message})
        return history
    except Exception as e:
        print(f"Error reading history: {e}")
        return []

def save_message(user_id, role, message):
    """Save a single message to the conversation memory sheet."""
    try:
        sheet = get_sheet()
        sheet.append_row([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            user_id,
            role,
            message
        ])
    except Exception as e:
        print(f"Error saving message: {e}")

# ── SLACK APP ────────────────────────────────────────────────
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

    logger.info(f"DM from {user_id}: {text}")

    # Save the incoming user message to memory
    save_message(user_id, "user", text)

    # Retrieve recent conversation history for this user
    history = get_history(user_id)

    # Build history string to pass to Zapier alongside the message
    history_text = ""
    if len(history) > 1:  # More than just the current message
        history_text = "\n\nCONVERSATION HISTORY (most recent last):\n"
        # Exclude the last message since it's the current one
        for msg in history[:-1]:
            prefix = "AE" if msg["role"] == "user" else "Buster"
            history_text += f"{prefix}: {msg['content']}\n"
        history_text += "\nCURRENT MESSAGE:\n"

    full_message = history_text + text

    try:
        response = requests.post(
            ZAPIER_WEBHOOK,
            params={"text": full_message, "user": user_id},
            timeout=30
        )
        # Try to extract Buster's response to save it to memory
        # Zapier handles the actual Slack response, but we log it
        if response.status_code == 200:
            logger.info(f"Zapier webhook triggered for {user_id}")
    except Exception as e:
        logger.error(f"Failed to reach Zapier: {e}")
        say("Sorry, I ran into an issue. Please try again in a moment.")

# ── HEALTH CHECK SERVER ──────────────────────────────────────
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Buster is running!")
    def log_message(self, format, *args):
        pass

def run_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    server.serve_forever()

if __name__ == "__main__":
    t = threading.Thread(target=run_health_server, daemon=True)
    t.start()
    print(f"Health server running on port {PORT}")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
