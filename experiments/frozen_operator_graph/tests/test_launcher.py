from pathlib import Path
import subprocess


def test_qa_launcher_contains_complete_server_paths_and_valid_shell():
    root = Path(__file__).resolve().parents[1]
    launcher = root / "run_qa.sh"
    text = launcher.read_text(encoding="utf-8")
    required = (
        "/share/home/tm902089733300000/a903202310/lys/research/graph",
        "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct",
        "/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl",
        "fresh_attention_c8847872bedf_20260731T074520Z_p876",
        "--source-json",
        "ROUTE_MASS_RETENTION=${ROUTE_MASS_RETENTION:-1.0}",
        "VALUE_ENERGY_RETENTION=${VALUE_ENERGY_RETENTION:-1.0}",
    )
    for value in required:
        assert value in text
    assert ': "${SPLIT_ROOT:?' not in text
    assert ': "${MODEL_PATH:?' not in text
    assert ': "${OUT:?' not in text
    result = subprocess.run(
        ["bash", "-n", str(launcher)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
