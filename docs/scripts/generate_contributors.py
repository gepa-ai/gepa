#!/usr/bin/env python3
"""
Generate the contributor list in README.md.

Run from anywhere (needs an authenticated ``gh`` CLI):

    uv run python docs/scripts/generate_contributors.py

Contributors are, merged into one alphabetical list:
- commit authors and ``Co-authored-by`` trailers on gepa-ai/gepa's default branch,
- authors of every post on the GEPA blog (docs/docs/blog),
- authors of the GEPA and optimize_anything papers (fetched from arXiv).

A name links to a GitHub account only when the handle is confirmed: the
account authored commits here, its GitHub profile name matches the person's
name, or it is listed in GITHUB_HANDLES below.
"""

import json
import re
import subprocess
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = "gepa-ai/gepa"
ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
BLOG = ROOT / "docs" / "docs" / "blog"
PAPERS = ("2507.19457", "2605.19633")  # GEPA, optimize_anything
START, END = "<!-- contributors:start -->", "<!-- contributors:end -->"

# Handles checked by hand, for people with no commits here whose GitHub
# profile name differs from the name they publish under (or is empty).
GITHUB_HANDLES = {
    "Alexandros G. Dimakis": "Alex-adim",
    "Alvin Cheung": "akcheung",
    "Arnav Singhvi": "arnavsinghvi11",
    "Christopher Potts": "cgpotts",
    "Devvrit Khatri": "devvrit",
    "Dilara Soylu": "dilarasoylu",
    "Herumb Shandilya": "krypticmouse",
    "Ion Stoica": "istoica",
    "Jialin Zhang": "Benzhang2004",
    "Koushik Sen": "ksen007",
    "Krista Opsahl-Ong": "Kopsahlong",
    "Meng Jiang": "mjiang89",
    "Michael J Ryan": "XenonMolecule",
    "Noah Ziems": "Ziems",
    "Qiuyang Mang": "joyemang33",
    "Rishi Khare": "rishiskhare",
    "Rodolfo Nobrega": "rodolfo-nobrega",
    "Roi Pony": "roipony",
    "Runyuan He": "runyuanhe",
    "Sanjit A. Seshia": "saseshia",
    "Tal Oved": "Tal-Oved",
}
# Secondary accounts, listed under the main one.
ALIASES = {"LakshyAAAgrawalP": "LakshyAAAgrawal"}
BOT_LOGINS = {"gepa-bot", "Copilot", "cursoragent", "ampagent", "actions-user"}
BOT_NAME = re.compile(r"^(claude|copilot|cursor|amp|hackmd|github actions|ubuntu)\b|\[bot\]", re.IGNORECASE)
NOREPLY = re.compile(r"^(?:\d+\+)?([^@]+)@users\.noreply\.github\.com$", re.IGNORECASE)
CO_AUTHOR = re.compile(r"^Co-authored-by:\s*(.+?)\s*<([^>]+)>\s*$", re.IGNORECASE | re.MULTILINE)

# Lower rank wins when choosing the name to show.
PAPER, BLOG_POST, GIT = 0, 1, 2


def _key(name: str) -> str:
    """Normalize a name for matching: drop parentheticals, accents, punctuation and case."""
    name = re.sub(r"\(.*?\)", " ", name)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


def _gh(*args: str) -> str:
    return subprocess.run(["gh", "api", *args], check=True, capture_output=True, text=True).stdout


def _github_user(login: str) -> dict | None:
    try:
        return json.loads(_gh(f"users/{login}"))
    except subprocess.CalledProcessError:
        return None


def commit_records() -> list[tuple[str, str | None, int]]:
    """(name, login, rank) for every commit author and co-author on the default branch."""
    lines = _gh(
        "--paginate",
        f"repos/{REPO}/commits?per_page=100",
        "--jq",
        ".[] | {login: .author.login, type: .author.type, name: .commit.author.name,"
        " email: .commit.author.email, message: .commit.message} | @json",
    ).splitlines()
    commits = [json.loads(line) for line in lines if line]
    for c in commits:
        c["human"] = c["type"] != "Bot" and c["login"] not in BOT_LOGINS and not BOT_NAME.search(c["name"])
    email_login = {c["email"].lower(): c["login"] for c in commits if c["login"] and c["human"]}

    records = []
    for c in commits:
        if c["human"]:
            records.append((c["name"], c["login"], GIT))
        for name, email in CO_AUTHOR.findall(c["message"]):
            if BOT_NAME.search(name):
                continue
            noreply = NOREPLY.match(email)
            login = noreply.group(1) if noreply else email_login.get(email.lower())
            records.append((name, login, GIT))
    return [(name, ALIASES.get(login, login) if login else None, rank) for name, login, rank in records]


def blog_records() -> tuple[list[tuple[str, str, int]], dict[str, str]]:
    """(name, "", rank) for every blog post author, and the GitHub handle each author entry gives."""
    authors = yaml.safe_load((BLOG / ".authors.yml").read_text(encoding="utf-8"))["authors"]
    used = set()
    for post in sorted((BLOG / "posts").rglob("*.md")):
        match = re.match(r"---\n(.*?)\n---\n", post.read_text(encoding="utf-8"), re.DOTALL)
        if match:
            used.update((yaml.safe_load(match.group(1)) or {}).get("authors") or [])

    records, handles = [], {}
    for key in sorted(used):
        entry = authors[key]
        name = entry["name"].split(",")[0].strip()
        records.append((name, "", BLOG_POST))
        for field in ("url", "avatar"):
            handle = re.match(r"https://github\.com/([A-Za-z0-9-]+)(?:\.png)?/?$", entry.get(field, ""))
            if handle:
                handles[name] = handle.group(1)
                break
    return records, handles


def paper_records() -> list[tuple[str, str, int]]:
    """(name, "", rank) for every author of the papers in PAPERS."""
    url = f"https://export.arxiv.org/api/query?id_list={','.join(PAPERS)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        feed = ET.fromstring(response.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    return [(n.text.strip(), "", PAPER) for n in feed.findall("a:entry/a:author/a:name", ns) if n.text]


def contributors() -> list[tuple[str, str | None]]:
    """(display name, GitHub login or None) per person, sorted by name."""
    git = commit_records()
    blog, blog_handles = blog_records()
    records = [*git, *blog, *paper_records()]
    committers = {login for _, login, _ in git if login}

    # Resolve a login for records that lack one: hand-checked handles first,
    # then a blog author's handle if it is confirmed.
    handles = {_key(name): login for name, login in GITHUB_HANDLES.items()}
    with ThreadPoolExecutor(max_workers=8) as pool:
        profiles = dict(zip(blog_handles.values(), pool.map(_github_user, blog_handles.values()), strict=True))
    for name, handle in blog_handles.items():
        profile = profiles[handle]
        if profile and profile["type"] == "Organization":
            records = [r for r in records if r[0] != name]
        elif profile and (
            profile["login"] in committers
            or profile["login"] in GITHUB_HANDLES.values()
            or _key(profile.get("name") or "") == _key(name)
        ):
            handles.setdefault(_key(name), profile["login"])
    lower_logins = {login.lower(): login for login in committers | set(handles.values())}
    records = [
        (name, login or handles.get(_key(name)) or lower_logins.get(name.lower()), rank)
        for name, login, rank in records
    ]

    # Group records that share a login or a normalized name (union-find).
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for name, login, _ in records:
        if login:
            parent[find(f"name:{_key(name)}")] = find(f"login:{login}")
        find(f"name:{_key(name)}")
    groups: dict[str, list[tuple[str, str | None, int]]] = {}
    for record in records:
        groups.setdefault(find(f"name:{_key(record[0])}"), []).append(record)

    people = []
    logins_needing_profile = set()
    for group in groups.values():
        logins = Counter(login for _, login, _ in group if login)
        login = logins.most_common(1)[0][0] if logins else None
        if len(logins) > 1:
            print(f"warning: {sorted(logins)} grouped as one person; linking {login}")
        # Prefer a paper or blog name, then a commit name that is not just the
        # login: full names (with a space) first, then the most used.
        names = Counter(
            (rank, name) for name, _, rank in group if rank < GIT or not (login and name.lower() == login.lower())
        )
        best = min(names, key=lambda rn: (rn[0], " " not in rn[1], -names[rn], -len(rn[1]))) if names else None
        people.append([best[1] if best else None, login])
        if best is None and login:
            logins_needing_profile.add(login)

    need = sorted(logins_needing_profile)
    with ThreadPoolExecutor(max_workers=8) as pool:
        profile_names = {
            login: (profile or {}).get("name")
            for login, profile in zip(need, pool.map(_github_user, need), strict=True)
        }
    for person in people:
        if person[0] is None:
            person[0] = profile_names.get(person[1]) or person[1]
    return sorted(((name, login) for name, login in people), key=lambda p: p[0].casefold())


def main() -> None:
    lines = [f"- [{name}](https://github.com/{login})" if login else f"- {name}" for name, login in contributors()]
    text = README.read_text(encoding="utf-8")
    start, end = text.index(START) + len(START), text.index(END)
    README.write_text(text[:start] + "\n" + "\n".join(lines) + "\n" + text[end:], encoding="utf-8")
    print(f"Wrote {len(lines)} contributors to {README.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
