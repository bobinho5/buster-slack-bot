import os
import threading
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ── CREDENTIALS ──────────────────────────────────────────────
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
ZAPIER_WEBHOOK  = os.environ["ZAPIER_WEBHOOK_URL"]
PORT            = int(os.environ.get("PORT", 8080))
# ─────────────────────────────────────────────────────────────

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

    try:
        requests.post(
            ZAPIER_WEBHOOK,
            params={"text": text, "user": user_id},
            timeout=10
        )
    except Exception as e:
        logger.error(f"Failed to reach Zapier: {e}")
        say("Sorry, I ran into an issue. Please try again in a moment.")

# Simple HTTP server so Render health checks pass
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
    # Run health check server in background thread
    t = threading.Thread(target=run_health_server, daemon=True)
    t.start()
    print(f"Health server running on port {PORT}")

    # Start Slack Socket Mode handler
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
