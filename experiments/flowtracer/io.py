import json
from pathlib import Path


def save_sample_graph(path, graph, analysis=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"graph": graph, "analysis": analysis or {}}, indent=2, ensure_ascii=False), encoding="utf-8")


def load_sample_graph(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["graph"], payload.get("analysis", {})
