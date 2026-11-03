"""
HistoScan - Breast Cancer Detection and Classification Using Deep Learning
Flask web application.   Run:  python app.py   then open http://127.0.0.1:5000
"""
import json, os, sqlite3, time, uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort
from PIL import Image, UnidentifiedImageError

from bcnet.predictor import Predictor, ROOT

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
DB = os.path.join(ROOT, "instance", "history.db")
UPLOADS = os.path.join(ROOT, "static", "uploads")
os.makedirs(os.path.dirname(DB), exist_ok=True); os.makedirs(UPLOADS, exist_ok=True)
predictor = Predictor()


def load_json(rel, default=None):
    p = os.path.join(ROOT, rel)
    return json.load(open(p)) if os.path.exists(p) else default


METRICS = load_json("models/idc_metrics.json", {})
SAMPLES = load_json("static/samples/samples.json", [])


def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id TEXT PRIMARY KEY, ts TEXT, filename TEXT, thumb TEXT, model TEXT, mode TEXT,
        label TEXT, prob REAL, confidence REAL, ms INTEGER, patient TEXT, notes TEXT)""")
    return con


TEAM = [("Harsh Siroya", "41"), ("Jinay Shah", "77"), ("Diyaan Singhvi", "39"), ("Yash Rana", "76")]


@app.context_processor
def inject():
    return {"team": TEAM, "guide": "Dr. Reena Lokare", "year": "2026-27", "nav": request.endpoint}


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html", m=METRICS, samples=SAMPLES[:6])


@app.route("/analyze")
def analyze():
    return render_template("analyze.html", backends=predictor.backends(), samples=SAMPLES)


@app.route("/model")
def model_page():
    return render_template("model.html", m=METRICS)


@app.route("/performance")
def performance():
    return render_template("performance.html", m=METRICS)


@app.route("/dataset")
def dataset():
    return render_template("dataset.html", m=METRICS, samples=SAMPLES)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/history")
def history():
    with db() as con:
        rows = con.execute("SELECT id,ts,filename,thumb,model,mode,label,prob,confidence,ms,patient "
                           "FROM predictions ORDER BY ts DESC LIMIT 200").fetchall()
    keys = ["id", "ts", "filename", "thumb", "model", "mode", "label", "prob", "confidence", "ms", "patient"]
    rows = [dict(zip(keys, r)) for r in rows]
    stats = {"n": len(rows), "mal": sum(r["label"] == "Malignant" for r in rows)}
    return render_template("history.html", rows=rows, stats=stats)


@app.route("/report/<rid>")
def report(rid):
    with db() as con:
        r = con.execute("SELECT id,ts,filename,thumb,model,mode,label,prob,confidence,ms,patient,notes "
                        "FROM predictions WHERE id=?", (rid,)).fetchone()
    if not r:
        abort(404)
    keys = ["id", "ts", "filename", "thumb", "model", "mode", "label", "prob", "confidence", "ms", "patient", "notes"]
    rec = dict(zip(keys, r))
    rec["detail"] = json.loads(rec.pop("notes") or "{}")
    return render_template("report.html", r=rec)


@app.post("/history/clear")
def clear_history():
    with db() as con:
        con.execute("DELETE FROM predictions")
    return redirect(url_for("history"))


# ---------------------------------------------------------------- API
@app.post("/api/predict")
def api_predict():
    backend = request.form.get("backend", "numpy")
    patient = (request.form.get("patient") or "").strip()[:60]
    f = request.files.get("image")
    sample = request.form.get("sample")
    try:
        if f and f.filename:
            img, name = Image.open(f.stream), f.filename
        elif sample and sample in {s["file"] for s in SAMPLES} | set(os.listdir(os.path.join(ROOT, "static", "samples"))):
            img, name = Image.open(os.path.join(ROOT, "static", "samples", sample)), sample
        else:
            return jsonify(error="Choose an image or a sample first."), 400
        img.load()
    except UnidentifiedImageError:
        return jsonify(error="That file is not an image. Upload a PNG, JPG or TIFF."), 400
    try:
        res = predictor.predict(img, backend)
    except RuntimeError as e:
        return jsonify(error=str(e)), 400

    rid = uuid.uuid4().hex[:10]
    thumb = img.convert("RGB"); thumb.thumbnail((96, 96))
    thumb_name = f"{rid}.png"; thumb.save(os.path.join(UPLOADS, thumb_name))
    detail = {k: res[k] for k in ("steps", "explain", "explain_title", "explain_note", "risk") if k in res}
    detail["tiles"] = res.get("tiles")
    with db() as con:
        con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, datetime.now().isoformat(timespec="seconds"), name, thumb_name, res["model"], res["mode"],
                     res["label"], res["prob"], res["confidence"], res["ms"], patient, json.dumps(detail)))
    res.update(id=rid, filename=name)
    return jsonify(res)


@app.get("/api/health")
def health():
    return jsonify(status="ok", backends=predictor.backends(), metrics_loaded=bool(METRICS))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  HistoScan running ->  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("DEBUG") == "1")
