import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ── CREDENTIALS ──────────────────────────────────────────────
SLACK_BOT_TOKEN   = os.environ["SLACK_BOT_TOKEN"]      # xoxb-...
SLACK_APP_TOKEN   = os.environ["SLACK_APP_TOKEN"]      # xapp-...
ZAPIER_WEBHOOK    = os.environ["ZAPIER_WEBHOOK_URL"]   # https://hooks.zapier.com/...
# ─────────────────────────────────────────────────────────────

app = App(token=SLACK_BOT_TOKEN)

@app.event("message")
def handle_dm(event, say, logger):
    # Only respond to direct messages (not channel messages, not bot messages)
    if event.get("channel_type") != "im":
        return
    if event.get("bot_id"):
        return

    user_id = event.get("user")
    text    = event.get("text", "").strip()

    if not text or not user_id:
        return

    logger.info(f"DM from {user_id}: {text}")

    # Forward to Zapier — Zapier calls Claude and posts response back via Slack API
    try:
        requests.post(
            ZAPIER_WEBHOOK,
            params={"text": text, "user": user_id},
            timeout=10
        )
    except Exception as e:
        logger.error(f"Failed to reach Zapier: {e}")
        say("Sorry, I ran into an issue. Please try again in a moment.")

if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
