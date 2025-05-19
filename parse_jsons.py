#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

# -------- helpers --------
def extract_json_objects(text: str):
    dec = json.JSONDecoder()
    i = 0
    while i < len(text):
        try:
            obj, end = dec.raw_decode(text, i)
            yield obj
            i = end
        except json.JSONDecodeError:
            i += 1

def strip_ws(s: str) -> str:
    return '\n'.join(l.rstrip() for l in s.splitlines() if l.strip())

# scenario → thinking / remainder
def split_scenario(resp: str):
    m = re.search(r'##\s*Scenario[\s\S]*?(?=\n##|\Z)', resp, re.I)
    if m:
        think = strip_ws(m.group(0))
        rest = resp.replace(m.group(0), '', 1)
        return think, strip_ws(rest)
    return "", strip_ws(resp)

CODE_RE = re.compile(r'```(\w*)\n([\s\S]*?)```')

def ensure_lang_token(segment: str, default: str = "python") -> str:
    def repl(m):
        lang = m.group(1) or default
        code = m.group(2)
        return f"```{lang}\n{code}```"
    return CODE_RE.sub(repl, segment)

def bold_topic(text: str, topic=r'Strategy Pattern'):
    return re.sub(topic, lambda m: f'**{m.group(0)}**', text, flags=re.I)

def code_blocks_only(text: str) -> str:
    blocks = CODE_RE.findall(text)
    out = []
    for lang, code in blocks:
        if not lang:
            lang = "python"
        out.append(f"```{lang}\n{code}```")
    return '\n\n'.join(out)

# -------- main sanitize --------
def sanitize(record: dict,
             context_key="context",
             user_key="user",
             think_key="assistant_thinking",
             assistant_key="assistant"):
    thinking_raw, assistant_raw = split_scenario(record.get("response", ""))

    thinking = bold_topic(ensure_lang_token(thinking_raw))
    assistant = code_blocks_only(ensure_lang_token(assistant_raw))

    return {
        context_key: strip_ws(record.get("system", "")),
        user_key: strip_ws(record.get("instruction", "")),
        think_key: thinking,
        assistant_key: assistant,
    }

# -------- CLI --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--context_key", default="context")
    parser.add_argument("--user_key", default="user")
    parser.add_argument("--thinking_key", default="assistant_thinking")
    parser.add_argument("--assistant_key", default="assistant")
    args = parser.parse_args()

    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in in_dir.rglob("*.json"):
        objs = list(extract_json_objects(fp.read_text(encoding="utf-8")))
        if not objs:
            continue
        out_file = out_dir / fp.name
        with out_file.open("w", encoding="utf-8") as w:
            for obj in objs:
                cleaned = sanitize(
                    obj,
                    context_key=args.context_key,
                    user_key=args.user_key,
                    think_key=args.thinking_key,
                    assistant_key=args.assistant_key,
                )
                json.dump(cleaned, w, ensure_ascii=False)
                w.write("\n")

if __name__ == "__main__":
    main()
