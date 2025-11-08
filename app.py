from flask import Flask, request, jsonify
from pathlib import Path
import threading
import json
from datetime import datetime
import shutil
import os
from typing import Optional, Tuple, Dict, Any

# disable TF oneDNN optimizations (keeps server logs cleaner)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import the processing components from the main module
from final_project_album_graph_creator import (
    ImageAnalyzer,
    CentralityGraphGenerator,
    ResultStorageParquet,
    setup_root_logger,
)

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

# Root folder for albums
UPLOAD_ROOT = Path("static") / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# Track background processing threads per album and last run timestamp
_processing_threads: dict[str, threading.Thread] = {}
_last_run_ts: dict[str, str] = {}


# -------------------- UI --------------------
@app.route("/")
def index():
    return app.send_static_file("UI_user_interface.html")


# -------------------- Albums --------------------
@app.route("/api/albums")
def list_albums():
    albums = [p.name for p in UPLOAD_ROOT.iterdir() if p.is_dir()]
    return jsonify(albums)


@app.route("/api/create_album", methods=["POST"])
def create_album():
    data = request.json or {}
    name = data.get("album", "").strip()
    if not name:
        return jsonify(error="No album name provided"), 400
    folder = UPLOAD_ROOT / name
    if folder.exists():
        return jsonify(error="Album already exists"), 400
    folder.mkdir(parents=True, exist_ok=True)
    return jsonify(''), 200


# -------------------- Upload --------------------
@app.route("/api/upload", methods=["POST"])
def upload_images():
    album = request.form.get("album", "").strip()
    if not album:
        return jsonify(error="Missing album"), 400

    folder = UPLOAD_ROOT / album
    if not folder.exists():
        return jsonify(error="Album not found"), 404

    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files uploaded"), 400

    for f in files:
        dest = folder / f.filename
        f.save(dest)

    return jsonify(''), 200


# -------------------- Processing --------------------
@app.route("/api/start_processing", methods=["POST"])
def start_processing():
    """
    Launch the full pipeline for the given album in a background thread.
    Outputs are stored per-run (timestamped) and copied into the album folder:
      - centrality_graph_<ts>.png
      - centrality_metrics_<ts>.json
      - graph_<ts>.json
    """
    data = request.json or {}
    album = data.get("album", "").strip()
    if not album:
        return jsonify(error="Missing album"), 400

    folder = UPLOAD_ROOT / album
    if not folder.exists():
        return jsonify(error="Album not found"), 404

    def background():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _last_run_ts[album] = ts

        listener, log_filename = setup_root_logger()
        try:
            analyzer = ImageAnalyzer(str(folder))
            analyzer.timestamp = ts
            analyzer.storage = ResultStorageParquet(ts)

            analyzer.process_images_parallel()
            analyzer.compare_all_images()

            gen = CentralityGraphGenerator(analyzer.storage, ts)
            graph_png_path = folder / f"centrality_graph_{ts}.png"
            metrics_dict = gen.create_graph(save_path=str(graph_png_path))

            if metrics_dict:
                (folder / f"centrality_metrics_{ts}.json").write_text(
                    json.dumps(metrics_dict, indent=2), encoding="utf-8"
                )

            graph_json_cwd = Path(f"graph_{ts}.json")
            if graph_json_cwd.exists():
                shutil.copyfile(graph_json_cwd, folder / graph_json_cwd.name)

        except Exception as e:
            app.logger.exception("Background processing failed: %s", e)
        finally:
            listener.stop()

    th = threading.Thread(target=background, daemon=True)
    th.start()
    _processing_threads[album] = th
    return jsonify(''), 200


@app.route("/api/status/<album>")
def status(album):
    th = _processing_threads.get(album)
    processing = bool(th and th.is_alive())
    return jsonify(processing=processing, last_ts=_last_run_ts.get(album))


# -------------------- Helpers --------------------
def _latest_file_in_album(album: str, pattern: str) -> Optional[Path]:
    """
    Return the most recently modified file matching `pattern` within the album folder.
    Example: 'centrality_metrics_*.json'
    """
    folder = UPLOAD_ROOT / album
    if not folder.exists():
        return None
    matches = list(folder.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _load_album_metrics(album: str) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """
    Load the latest centrality metrics JSON for the album.
    """
    latest = _latest_file_in_album(album, "centrality_metrics_*.json")
    if not latest or not latest.exists():
        return None, None
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        app.logger.error(f"Failed to read metrics file {latest}: {e}")
        return None, None
    return data, latest



# -------------------- Centrality APIs --------------------
@app.route("/api/top_central/<album>")
def top_central(album):
    """
    Return centrality scores for the given album and metric.
    Query params:
      - metric: one of {degree, closeness, harmonic, eigenvector, pagerank, betweenness}
                default: eigenvector
      - n: integer count OR the string 'all' (case-insensitive)
           * if omitted or invalid -> return ALL files (default behavior)
           * if 'all'            -> return ALL files
           * if integer          -> return top-N

    Response:
      {
        "album": "...",
        "metric": "...",
        "count": <number of items>,
        "source": "centrality_metrics_<ts>.json",
        "results": [
          {"filename": "<file>", "score": <float>},
          ...
        ]
      }
    """
    metric = request.args.get("metric", "eigenvector").strip().lower()
    n_param = request.args.get("n", "").strip()

    metrics, path = _load_album_metrics(album)
    if metrics is None:
        return jsonify(error="No metrics found for album"), 404

    if metric not in metrics:
        return jsonify(error=f"Unknown metric '{metric}'", available=list(metrics.keys())), 400

    folder = UPLOAD_ROOT / album

    # Filter to files that physically exist in the album
    scores_dict: dict[str, float] = metrics[metric]
    items = [(fn, sc) for fn, sc in scores_dict.items() if (folder / fn).exists()]
    items.sort(key=lambda kv: kv[1], reverse=True)

    # Decide how many to return
    if n_param == "" or n_param.lower() == "all":
        limited = items  # default: ALL entries, sorted desc
    else:
        try:
            n = max(0, int(n_param))
            limited = items[:n]
        except ValueError:
            # If n is invalid, return ALL (default behavior)
            limited = items

    results = [{"filename": fn, "score": sc} for fn, sc in limited]

    return jsonify(
        album=album,
        metric=metric,
        count=len(results),
        source=str(path.name),
        results=results
    )


# Backward-compatible endpoint: top-10 eigenvector filenames only
@app.route("/api/central_images/<album>")
def central_images(album):
    metric = "eigenvector"
    # Ask our own endpoint for top-10
    resp = app.test_client().get(f"/api/top_central/{album}?metric={metric}&n=10")
    if resp.status_code != 200:
        return resp
    data = resp.get_json()
    return jsonify(central_images=[r["filename"] for r in data["results"]])


@app.route("/api/metrics/<album>")
def metrics_endpoint(album):
    metrics_dict, path = _load_album_metrics(album)
    if metrics_dict is None:
        return jsonify(error="No metrics found for album"), 404
    return jsonify(
        album=album,
        available_metrics=list(metrics_dict.keys()),
        source=str(path.name),
        metrics=metrics_dict
    )


@app.route("/api/graph/<album>")
def graph_json(album):
    """
    Return the latest graph_<ts>.json for the album if present.
    """
    latest = _latest_file_in_album(album, "graph_*.json")
    if not latest:
        return jsonify(error="Graph JSON not found"), 404
    data = json.loads(latest.read_text(encoding="utf-8"))
    return jsonify(data)


# -------------------- Tags --------------------
@app.route("/api/save_tags", methods=["POST"])
def save_tags():
    """
    Expect: {"album": "...", "tags": [...]}
    Writes to: static/central_images_tags_<album>.json
    """
    data = request.json or {}
    album = data.get("album", "").strip()
    tags = data.get("tags")

    if not album or tags is None:
        return jsonify(error="Missing album or tags"), 400

    # Ensure the static root exists
    static_root = Path(app.static_folder)
    static_root.mkdir(parents=True, exist_ok=True)

    # Save in static root
    out_name = f"central_images_tags_{album}.json"
    out_path = static_root / out_name
    out_path.write_text(json.dumps(tags, indent=2, ensure_ascii=False), encoding="utf-8")

    return jsonify(message="saved", path=f"/{out_name}"), 200


# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
