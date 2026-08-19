import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "experiments" / "rr_topology_dynamics" / "run.sh"


def _bash_executable() -> str | None:
    discovered = shutil.which("bash")
    if discovered:
        return discovered
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    return str(git_bash) if git_bash.is_file() else None


FAKE_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >> "$CALL_LOG"

output=""
output_dir=""
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[$index]}" in
    --output)
      output="${arguments[$((index + 1))]}"
      ;;
    --output-dir)
      output_dir="${arguments[$((index + 1))]}"
      ;;
  esac
done

if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  printf 'artifact\n' > "$output"
fi
if [[ -n "$output_dir" ]]; then
  mkdir -p "$output_dir"
  printf '{}\n' > "$output_dir/report.json"
fi

printf 'fake stage complete\n'
"""


class TopologyRunScriptTest(unittest.TestCase):
    def test_clean_run_builds_spectral_reference_then_completes_topology(self):
        bash = _bash_executable()
        if bash is None:
            self.skipTest("bash is unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "train").mkdir(parents=True)
            (data / "test").mkdir(parents=True)
            (data / "train" / "manifest.json").write_text("{}\n", encoding="utf-8")
            (data / "test" / "manifest.json").write_text("{}\n", encoding="utf-8")

            fake_python = root / "fake_python.sh"
            fake_python.write_text(FAKE_PYTHON, encoding="utf-8", newline="\n")
            fake_python.chmod(0o755)

            call_log = root / "calls.log"
            spectral_reference = root / "spectral" / "reference.npz"
            output = root / "topology"
            environment = {
                **os.environ,
                "ROOT": data.as_posix(),
                "PYTHON": fake_python.as_posix(),
                "DEVICE": "cpu",
                "LIMIT": "1",
                "RUN_TESTS": "0",
                "SPECTRAL_REFERENCE": spectral_reference.as_posix(),
                "OUT": output.as_posix(),
                "CALL_LOG": call_log.as_posix(),
            }

            completed = subprocess.run(
                [bash, RUN_SCRIPT.as_posix()],
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 4)
            self.assertIn("experiments.spectral_feasibility.main fit", calls[0])
            self.assertIn("--limit 32", calls[0])
            self.assertIn("experiments.rr_topology_dynamics.main fit", calls[1])
            self.assertIn("--limit 1", calls[1])
            self.assertIn("experiments.rr_topology_dynamics.main score", calls[2])
            self.assertIn("experiments.rr_topology_dynamics.main evaluate", calls[3])
            self.assertTrue(spectral_reference.is_file())
            self.assertTrue((output / "reference.npz").is_file())
            self.assertTrue((output / "test_features.npz").is_file())
            self.assertTrue((output / "evaluation" / "report.json").is_file())
            self.assertTrue((output / "logs" / "spectral_fit.log").is_file())
            self.assertTrue((output / "logs" / "topology_fit.log").is_file())
            self.assertTrue((output / "logs" / "topology_score.log").is_file())
            self.assertTrue((output / "logs" / "topology_evaluate.log").is_file())


if __name__ == "__main__":
    unittest.main()
