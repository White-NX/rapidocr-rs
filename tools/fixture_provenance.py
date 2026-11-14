from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

RAPIDOCR_PYTHON_REPO_ENV = "RAPIDOCR_PYTHON_REPO"
PROVENANCE_FIELDS = (
    "source",
    "source_commit",
    "source_package_version",
    "model_version",
)


@dataclass(frozen=True)
class FixtureProvenance:
    source: str
    source_commit: str
    source_package_version: str
    model_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "source_commit": self.source_commit,
            "source_package_version": self.source_package_version,
            "model_version": self.model_version,
        }


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def python_package_version(repo: Path) -> str:
    describe = git_output(repo, "describe", "--tags", "--always")
    tag = git_output(repo, "describe", "--tags", "--abbrev=0")
    if describe == tag:
        return tag.removeprefix("v")
    version = tag.removeprefix("v")
    distance, _revision = describe.removeprefix(tag + "-").split("-", 1)
    release = [int(part) for part in version.split(".")]
    release[-1] += 1
    return f"{'.'.join(map(str, release))}.dev{distance}"


def model_version(repo: Path) -> str:
    model_file = repo / "python" / "rapidocr" / "default_models.yaml"
    revisions: set[str] = set()
    for line in model_file.read_text(encoding="utf-8").splitlines():
        marker = "/resolve/"
        if marker not in line or "/onnx/" not in line:
            continue
        revision = line.split(marker, 1)[1].split("/", 1)[0]
        revisions.add(revision)
    if not revisions:
        raise RuntimeError(f"no model revisions found in {model_file}")
    if len(revisions) == 1:
        return next(iter(revisions))
    return ",".join(sorted(revisions))


def read_provenance(repo: Path) -> FixtureProvenance:
    repo = repo.resolve()
    dirty = git_output(repo, "status", "--porcelain")
    if dirty:
        raise RuntimeError(
            f"Python RapidOCR repo {repo} has uncommitted changes; commit or stash them before exporting fixtures"
        )
    return FixtureProvenance(
        source="python-rapidocr",
        source_commit=git_output(repo, "rev-parse", "HEAD"),
        source_package_version=python_package_version(repo),
        model_version=model_version(repo),
    )


def fixture_files(rs_root: Path) -> list[Path]:
    candidates = sorted((rs_root / "fixtures" / "e2e").glob("*.json"))
    candidates.extend(sorted((rs_root / "fixtures" / "db_postprocess").glob("*/expected.json")))
    return [
        path
        for path in candidates
        if json.loads(path.read_text(encoding="utf-8")).get("source")
        == "python-rapidocr"
    ]


def update_fixtures(rs_root: Path, provenance: FixtureProvenance) -> int:
    files = fixture_files(rs_root)
    for path in files:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        newline = "\r\n" if b"\r\n" in raw else "\n"
        had_final_newline = text.endswith(("\n", "\r"))
        payload = json.loads(text)
        payload = {
            **provenance.as_dict(),
            **{
                key: value
                for key, value in payload.items()
                if key not in PROVENANCE_FIELDS
            },
        }
        updated = json.dumps(payload, ensure_ascii=False, indent=2).replace(
            "\n", newline
        )
        if had_final_newline:
            updated += newline
        path.write_bytes(updated.encode("utf-8"))
    return len(files)


def check_fixtures(rs_root: Path, provenance: FixtureProvenance) -> int:
    files = fixture_files(rs_root)
    expected = provenance.as_dict()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = {field: payload.get(field) for field in PROVENANCE_FIELDS}
        if actual != expected:
            raise RuntimeError(
                f"fixture provenance mismatch in {path}: expected {expected}, got {actual}"
            )
    return len(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--python-repo",
        type=Path,
        default=None,
        help=f"Path to the Python RapidOCR repo. Defaults to ${RAPIDOCR_PYTHON_REPO_ENV}.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite fixture JSON metadata. Without this flag, validate only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_repo = args.python_repo
    if python_repo is None:
        value = os.environ.get(RAPIDOCR_PYTHON_REPO_ENV)
        if value:
            python_repo = Path(value)
    if python_repo is None:
        raise SystemExit(
            f"--python-repo or ${RAPIDOCR_PYTHON_REPO_ENV} is required"
        )

    provenance = read_provenance(python_repo)
    rs_root = args.rs_root.resolve()
    count = (
        update_fixtures(rs_root, provenance)
        if args.update
        else check_fixtures(rs_root, provenance)
    )
    action = "updated" if args.update else "validated"
    print(f"{action} {count} fixtures with {provenance.as_dict()}")


if __name__ == "__main__":
    main()
