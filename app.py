import os
import traceback
from flask import Flask, request, jsonify, render_template, send_file

from ml_engine import AutoMLSession

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap

# In-memory session store: session_id -> AutoMLSession
# Fine for a single-process local/demo deployment.
SESSIONS = {}


def get_session(session_id):
    session = SESSIONS.get(session_id)
    if session is None:
        raise KeyError("Unknown or expired session_id. Upload a dataset again.")
    return session


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are supported right now."}), 400

    session = AutoMLSession()
    try:
        summary = session.load_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    SESSIONS[session.id] = session
    return jsonify(summary)


@app.route("/api/configure", methods=["POST"])
def configure():
    data = request.get_json(force=True)
    try:
        session = get_session(data.get("session_id"))
        result = session.configure(
            feature_columns=data.get("feature_columns", []),
            target_column=data.get("target_column"),
            task_type=data.get("task_type", "auto"),
        )
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


@app.route("/api/train", methods=["POST"])
def train():
    data = request.get_json(force=True)
    try:
        session = get_session(data.get("session_id"))
        epochs = int(data.get("epochs", 30))
        lr = float(data.get("lr", 0.001))
        batch_size = int(data.get("batch_size", 32))
        test_size = float(data.get("test_size", 0.2))
        epochs = max(1, min(epochs, 300))
        batch_size = max(1, min(batch_size, 512))
        test_size = min(max(test_size, 0.05), 0.5)

        result = session.train(test_size=test_size, epochs=epochs, lr=lr, batch_size=batch_size)
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Print the full traceback to the server terminal - if training
        # ever silently kills the connection again, this is where the real
        # cause will show up (the browser only ever sees "Failed to fetch").
        traceback.print_exc()
        return jsonify({"error": f"Training failed: {e}"}), 500

    return jsonify({
        "log": result["log"],
        "final": result["final"],
        "problem_mode": result["problem_mode"],
        "class_names": session.class_names,
        "feature_columns": session.feature_columns,
        "categorical_features": session.categorical_features,
        "categories": {c: list(session.cat_encoders[c].classes_) for c in session.categorical_features},
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        session = get_session(data.get("session_id"))
        if session.model is None:
            return jsonify({"error": "Train a model before predicting."}), 400
        result = session.predict_one(data.get("features", {}))
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500
    return jsonify(result)


@app.route("/api/download_model/<session_id>", methods=["GET"])
def download_model(session_id):
    try:
        session = get_session(session_id)
        if session.model is None:
            return jsonify({"error": "Train a model before downloading."}), 400
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    buf = session.to_bytes()
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"autoai_model_{session_id}.pkl",
        mimetype="application/octet-stream",
    )


@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True)
    SESSIONS.pop(data.get("session_id"), None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # use_reloader=False is important: the debug reloader watches files and
    # can restart the whole process mid-request (killing the socket the
    # browser is waiting on), which shows up in the browser as a generic
    # "Failed to fetch" with no error detail. threaded=True lets the static
    # file requests (css/js) keep being served while a long train request
    # is in flight.
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False, threaded=True)