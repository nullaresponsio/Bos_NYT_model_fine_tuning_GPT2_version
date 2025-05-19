#!/usr/bin/env python3
import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import requests
import wikipedia

wikipedia.set_lang("en")

CITATION_TOKEN = "<CITATION>"
SECTION_TOKEN = "<WIKIPEDIA_SECTION>"
WIKIPEDIA_LINK_TOKEN = "<WIKIPEDIA_LINK>"

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\[[^\]]*\]", "", t)
    return t.strip()

def strip_ws(s: str) -> str:
    return '\n'.join(l.rstrip() for l in s.splitlines() if l.strip())

def pseudo_random_thinking(seed_topic: str) -> str:
    starters = [
        f"Analyzing historical context for {seed_topic}.",
        f"Evaluating key events related to {seed_topic}.",
        f"Considering the implications and significance of {seed_topic}.",
        f"Compiling essential facts about {seed_topic}.",
        f"Exploring different perspectives regarding {seed_topic}.",
        f"Formulating a concise yet comprehensive summary of {seed_topic}.",
    ]
    return random.choice(starters)

def is_wikipedia_link(url: str) -> bool:
    return "wikipedia.org" in url

def fetch_page(title: str):
    try:
        return wikipedia.page(title, auto_suggest=False, redirect=True, preload=True)
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        return None

def extract_sections(content: str) -> str:
    sections, current = [], []
    for line in content.splitlines():
        if re.match(r'^=+[^=]+=+$', line.strip()):
            if current:
                sections.append('\n'.join(current).strip())
                current = []
            sections.append(f"{SECTION_TOKEN} {line.strip().strip('= ').strip()} {SECTION_TOKEN}")
        else:
            current.append(line)
    if current:
        sections.append('\n'.join(current).strip())
    return '\n\n'.join(s for s in sections if s.strip())

def build_assistant_response(content: str, url: str) -> str:
    if is_wikipedia_link(url):
        return f"{content}\n\n{WIKIPEDIA_LINK_TOKEN} {url}"
    return f"{content}\n\n{CITATION_TOKEN} {url}"

def fetch_wikipedia_sections(title: str) -> List[str]:
    page = fetch_page(title)
    return [s.strip() for s in page.sections if s.strip()] if page else []

def annotate_tokens(text: str) -> str:
    def wiki_repl(m):
        url = m.group(1)
        title = url.split("/")[-1].replace('_', ' ')
        secs = fetch_wikipedia_sections(title)
        sec_tokens = ''.join(f"\n{SECTION_TOKEN} {sec}" for sec in secs)
        return f"{WIKIPEDIA_LINK_TOKEN} {url}{sec_tokens}"
    def cite_repl(m):
        return f"{CITATION_TOKEN} {m.group(1)}"
    text = re.sub(rf"{WIKIPEDIA_LINK_TOKEN}\s*(https?://en\.wikipedia\.org/wiki/[^\s]+)", wiki_repl, text)
    text = re.sub(rf"{CITATION_TOKEN}\s*(https?://\S+)", cite_repl, text)
    return text

def fetch_citation(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        return clean_text(r.text)
    except Exception:
        return ""

def extract_external_links(html: str) -> List[str]:
    return [u for u in re.findall(r'https?://[^\s"\'\)]+', html) if not is_wikipedia_link(u)]

def process_citation(url: str, out_dir: Path) -> List[str]:
    content = fetch_citation(url)
    if not content:
        return []
    rec = {
        "context": url,
        "user": f"Fetch content from {url}",
        "assistant_thinking": "Retrieving and cleaning citation content.",
        "assistant": content,
    }
    fname = re.sub(r"[^\w\-]", "_", url)[:80] + ".json"
    (out_dir / fname).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return extract_external_links(content)

def dfs_citations(start_urls: List[str], out_dir: Path, depth_limit: int = 2):
    visited, stack = set(), [(u, 0) for u in start_urls]
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {}
        while stack or futures:
            while stack:
                url, depth = stack.pop()
                if url in visited or depth > depth_limit:
                    continue
                visited.add(url)
                futures[ex.submit(process_citation, url, out_dir)] = depth
            for fut in list(as_completed(futures)):
                depth = futures.pop(fut)
                for link in fut.result():
                    if link not in visited:
                        stack.append((link, depth + 1))

def sanitize(rec: dict) -> dict or None:
    content, topic, url = rec["response"], rec["system"], rec["url"]
    paras = [p for p in content.split('\n\n') if len(p.strip()) > 50]
    if len(paras) < 2 or sum(len(p.split()) for p in paras) < 250:
        page = fetch_page(topic)
        if not page:
            return None
        content = extract_sections(clean_text(page.content))
        url = page.url
        if len(content.split()) < 250:
            return None
    thinking = pseudo_random_thinking(topic)
    assistant = build_assistant_response(content, url)
    return {
        "context": strip_ws(topic),
        "user": f"Explain {topic}",
        "assistant_thinking": thinking,
        "assistant": assistant,
    }

def process_article(title: str, out_dir: Path):
    page = fetch_page(title)
    if not page:
        return
    rec = {
        "system": page.title,
        "response": clean_text(page.summary),
        "url": page.url,
    }
    for link in page.links[:3]:
        sub = fetch_page(link)
        if sub:
            rec["response"] += "\n\n" + clean_text(sub.summary)
            break
    out = sanitize(rec)
    if not out:
        return
    fname = re.sub(r"[^\w\-]", "_", title) + ".json"
    (out_dir / fname).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

def bfs_wikipedia(seeds: List[str], out_dir: Path, max_depth: int = 2, max_pages: int = 1000):
    visited = set()
    queue = [(s, 0) for s in seeds if s]
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as ex:
        while queue and len(visited) < max_pages:
            title, depth = queue.pop(0)
            if title in visited or depth > max_depth:
                continue
            visited.add(title)
            ex.submit(process_article, title, out_dir)
            page = fetch_page(title)
            if not page:
                continue
            for link in page.links:
                if link not in visited:
                    queue.append((link, depth + 1))

def process_json(in_path: str, out_path: str, citation_dir: Path, wiki_dir: Path, depth_limit: int = 2, wiki_depth: int = 2):
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    assistant = annotate_tokens(data.get("assistant", ""))
    data["assistant"] = assistant
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    citations = re.findall(rf"{CITATION_TOKEN}\s*(https?://\S+)", assistant)
    wiki_urls = re.findall(rf"{WIKIPEDIA_LINK_TOKEN}\s*(https?://en\.wikipedia\.org/wiki/[^\s]+)", assistant)
    wiki_titles = [u.split("/")[-1].replace("_", " ") for u in wiki_urls]
    dfs_citations(list(set(citations)), citation_dir, depth_limit)
    if wiki_titles:
        bfs_wikipedia(wiki_titles, wiki_dir, wiki_depth)

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_json", help="JSON file to transform")
    group.add_argument("--seeds", help='Comma-separated Wikipedia seed titles e.g. "Donald Trump,Barack Obama"')
    group.add_argument("--citation_urls", help='Comma-separated start URLs for citation crawl')
    parser.add_argument("--output_json", help="output path when --input_json is used")
    parser.add_argument("--citation_folder", default="citation_jsons")
    parser.add_argument("--wikipedia_folder", default="wikipedia_jsons")
    parser.add_argument("--depth_limit", type=int, default=2)
    parser.add_argument("--wiki_depth", type=int, default=2)
    parser.add_argument("--max_pages", type=int, default=1000)
    args = parser.parse_args()

    if args.input_json:
        if not args.output_json:
            parser.error("--output_json required with --input_json")
        t0 = time.time()
        process_json(
            args.input_json,
            args.output_json,
            Path(args.citation_folder),
            Path(args.wikipedia_folder),
            args.depth_limit,
            args.wiki_depth,
        )
        print(f"process_json completed in {time.time()-t0:.2f}s")

    elif args.seeds:
        seed_list = [s.strip().strip('"').strip("'") for s in args.seeds.split(",") if s.strip()]
        if not seed_list:
            parser.error("Provide at least one valid seed title in --seeds")
        t0 = time.time()
        bfs_wikipedia(
            seed_list,
            Path(args.wikipedia_folder),
            args.wiki_depth,
            args.max_pages,
        )
        print(f"Wikipedia crawl completed in {time.time()-t0:.2f}s")

    else:  # citation_urls
        url_list = [u.strip() for u in args.citation_urls.split(",") if u.strip()]
        if not url_list:
            parser.error("Provide at least one valid URL in --citation_urls")
        t0 = time.time()
        dfs_citations(url_list, Path(args.citation_folder), args.depth_limit)
        print(f"Citation crawl completed in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
