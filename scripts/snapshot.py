#!/usr/bin/env python3
"""Freeze every entry at the deadline: clone each entrant's repository, record exactly which commit was taken, and
lay the submissions out for scripts/grade_all.sh.

    python3 scripts/snapshot.py entries.tsv snapshots/

entries.tsv has one entry per line, `handle<TAB>repo URL` (lines starting with # are ignored), e.g.

    some_handle	https://github.com/someone/my-sprout-entry

Run it right at the deadline (30 Nov 2026, 23:59 IST = 18:29 UTC). Git commit dates can be set to anything, so the
commit that counts is the one this script fetched, at the time it ran; the manifest records both. It writes:

    snapshots/repos/<handle>/          the full clone (kept as evidence)
    snapshots/submissions/<handle>/    the submission folder only, ready for grade_all.sh
    snapshots/manifest.tsv             handle, url, status, commit, files hash, snapshot time (UTC), folder used

The submission folder is the repository root if it holds controller.py, otherwise the one folder in the repository
that holds a controller.py; anything else is reported, not guessed. Handles become folder (and container) names, so
they are reduced to letters, digits, '.', '_' and '-'.

Needs only git and Python 3. Uses the network (it is the one step that does); grading never does.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

CLONE_TIMEOUT = 300          # seconds per repository
MAX_MB = 200                 # a submission folder bigger than this is flagged, not copied


def safe(handle: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", handle.strip().lstrip("@"))
    return re.sub(r"^[^A-Za-z0-9]+", "", s)[:60] or "entry"


def files(folder: Path):
    """Regular files only. Symlinks are never followed: a link to the organiser's own files (arena.yaml, keys)
    would otherwise be copied into the submission and become readable by the entry."""
    return sorted(q for q in folder.rglob("*") if q.is_file() and not q.is_symlink() and ".git" not in q.parts)


def no_links(src, names):
    """Skip .git, symlinks, and compiled Python: a bundled .pyc can run code that differs from the .py we review."""
    return [n for n in names if n in (".git", "__pycache__") or n.endswith((".pyc", ".pyo")) or Path(src, n).is_symlink()]


def tree_hash(folder: Path) -> str:
    """sha256 over every file's relative path and contents, so the graded files can be checked later."""
    h = hashlib.sha256()
    for p in files(folder):
        h.update(str(p.relative_to(folder)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def find_submission(repo: Path) -> tuple[Path | None, str]:
    if (repo / "controller.py").is_file() and not (repo / "controller.py").is_symlink():
        return repo, "."
    found = [p.parent for p in repo.rglob("controller.py") if ".git" not in p.parts and not p.is_symlink()]
    if len(found) == 1:
        return found[0], str(found[0].relative_to(repo))
    return None, f"{len(found)} controller.py files found" if found else "no controller.py"


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          timeout=CLONE_TIMEOUT, env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"}).stdout.strip()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("\n\n")[1])
    entries, out = Path(sys.argv[1]), Path(sys.argv[2])
    repos, subs = out / "repos", out / "submissions"
    repos.mkdir(parents=True, exist_ok=True)
    subs.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for line in entries.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        handle, _, url = line.partition("\t")
        name, url = safe(handle), url.strip()
        if name in seen:
            rows.append((name, url, "DUPLICATE handle (first one kept)", "", "", "", ""))
            continue
        seen.add(name)
        when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        dest = repos / name
        shutil.rmtree(dest, ignore_errors=True)
        try:
            if not re.match(r"^(https://|file://|/)", url):
                raise ValueError("not an https:// repository URL")
            git("clone", "--quiet", "--no-recurse-submodules", url, str(dest))
            try:
                commit = git("rev-parse", "HEAD", cwd=dest)
            except subprocess.CalledProcessError:    # the remote's HEAD names a branch that doesn't exist
                branches = git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin", cwd=dest).split()
                branches = [b for b in branches if b != "origin/HEAD"]
                pick = next((b for b in ("origin/main", "origin/master") if b in branches), branches[0] if branches else None)
                if pick is None:
                    raise ValueError("the repository has no commits")
                git("checkout", "--quiet", "-B", pick.split("/", 1)[1], pick, cwd=dest)
                commit = git("rev-parse", "HEAD", cwd=dest)
        except Exception as e:
            msg = getattr(e, "stderr", "") or str(e)
            lines = [ln for ln in str(msg).strip().splitlines() if ln.strip()]
            first = next((ln for ln in lines if ln.startswith("fatal")), lines[0] if lines else str(e))
            rows.append((name, url, f"CLONE FAILED: {first}"[:200], "", "", when, ""))
            print(f"  {name}: clone failed")
            continue
        folder, where = find_submission(dest)
        if folder is None:
            rows.append((name, url, f"NO SUBMISSION: {where}", commit, "", when, ""))
            print(f"  {name}: {where}")
            continue
        size = sum(p.stat().st_size for p in files(folder)) / 1e6
        if size > MAX_MB:
            rows.append((name, url, f"TOO BIG: {size:.0f} MB", commit, "", when, where))
            print(f"  {name}: {size:.0f} MB, not copied")
            continue
        target = subs / name
        shutil.rmtree(target, ignore_errors=True)
        links = [str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_symlink()]
        shutil.copytree(folder, target, ignore=no_links, symlinks=True)
        lfs = [p.name for p in files(target) if p.stat().st_size < 200 and p.read_bytes().startswith(b"version https://git-lfs")]
        status = "OK" + (f"; symlinks dropped: {links[:3]}" if links else "") + (f"; Git LFS files not downloaded: {lfs[:3]}" if lfs else "")
        rows.append((name, url, status, commit, tree_hash(target), when, where))
        print(f"  {name}: {commit[:10]} ({where})")
    with open(out / "manifest.tsv", "w") as f:
        f.write("handle\turl\tstatus\tcommit\tfiles_sha256\tsnapshot_utc\tfolder\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    ok = sum(r[2].startswith("OK") for r in rows)
    print(f"{ok}/{len(rows)} entries ready in {subs}; manifest: {out / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
