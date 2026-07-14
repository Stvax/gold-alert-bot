import os
from flask import Flask, request
from gold_alert_bot import check_signal

app = Flask(__name__)


@app.route("/")
def home():
    return "Gold Alert Bot is running. Visit /check to run a check, or /check?test=true to send a test message."


@app.route("/check")
def check():
    test_mode = request.args.get("test", "false").lower() == "true"
    result = check_signal(test_mode)
    return result


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
