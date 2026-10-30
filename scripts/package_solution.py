"""Package only explicit handover files; secrets and machine state are excluded."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = ["README.md", "HANDOVER.md", "requirements.txt", ".env.example", ".gitignore",
             ".dockerignore", ".python-version", "Dockerfile", "pytest.ini", ".streamlit/config.toml"]
FOLDERS = ["app", "src", "tests", "data/raw", "data/processed", "artifacts", "docs", "scripts"]
SUFFIXES = {".py", ".csv", ".json", ".joblib", ".pdf", ".md", ".txt"}


def main():
    paths = [ROOT / name for name in TOP_LEVEL]
    for folder in FOLDERS:
        paths.extend(p for p in (ROOT / folder).rglob("*")
            if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts)
    paths = sorted(set(paths), key=lambda p: p.relative_to(ROOT).as_posix())
    assert all(p.exists() for p in paths)
    assert len(list((ROOT / "docs").glob("*.pdf"))) == 2
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out = ROOT / "dist" / "Credit_Risk_Portfolio_Solution.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in paths:
            z.write(p, "credit_risk_portfolio/" + p.relative_to(ROOT).as_posix())
        z.writestr("credit_risk_portfolio/MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert not any(".venv/" in name or name.endswith("/.env") or name.endswith("/secrets.toml") or "runtime/" in name for name in z.namelist())
    print(json.dumps({"path": str(out), "files": len(paths)+1, "bytes": out.stat().st_size,
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
