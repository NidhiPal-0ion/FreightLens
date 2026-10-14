"""Three untouched runs in three separate processes; compare the outputs byte for byte.

    python -m src.repro_check      -> logs/repro_check.txt
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("results.csv", "weekly_metrics.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lines = ["Reproducibility check: 3 independent runs (fresh process each) on the same input.",
             f"Input: {ROOT / 'data'}", ""]
    hashes = {f: [] for f in FILES}
    with tempfile.TemporaryDirectory() as tmp:
        for i in (1, 2, 3):
            out = Path(tmp) / f"run{i}"
            subprocess.run([sys.executable, "-m", "src.pipeline", "--out-dir", str(out / "outputs"),
                            "--logs-dir", str(out / "logs")], cwd=ROOT, check=True, capture_output=True)
            for f in FILES:
                hashes[f].append(sha256(out / "outputs" / f))
    ok = True
    for f in FILES:
        same = len(set(hashes[f])) == 1
        ok &= same
        lines.append(f"{f}: " + ("IDENTICAL across all 3 runs" if same else "DIFFERENT"))
        for i, h in enumerate(hashes[f], 1):
            lines.append(f"  run{i} sha256 {h}")
    shipped = sha256(ROOT / "outputs" / "results.csv")
    same_as_shipped = shipped == hashes["results.csv"][0]
    ok &= same_as_shipped
    lines += ["", "Submitted outputs/results.csv matches the runs: " + ("yes" if same_as_shipped else "NO"),
              "", "RESULT: " + ("PASS - flags, numbers, cited notes and wording are all identical."
                                if ok else "FAIL")]
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "repro_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
