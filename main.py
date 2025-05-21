import os
import re
import sys
import subprocess
import asyncio
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

# ──────────── CONFIGURATION ────────────
URL        = "https://ai.pydantic.dev/llms-full.txt"
OUTPUT_DIR = "out_chunks"
SUBFOLDER  = None
MODEL      = "gemma3:1b-it-qat"
MAX_CHARS  = 2000
OVERLAP    = 200
# ───────────────────────────────────────

def ensure_model(model: str):
    """
    Ensure `model` is pulled locally. Uses `ollama list` and `ollama pull`.
    """
    # list existing models
    lst = subprocess.run(
        ["ollama", "list", "models"],
        capture_output=True, text=True
    )
    if lst.returncode != 0:
        print("❗️ failed to run `ollama list models` – is Ollama installed?")
        sys.exit(1)
    if model not in lst.stdout:
        print(f"⬇️ Pulling model {model}…")
        pull = subprocess.run(["ollama", "pull", model])
        if pull.returncode != 0:
            print(f"❗️ failed to pull {model}")
            sys.exit(1)

def summarize_chunk(chunk: str, model: str) -> str:
    """
    Uses `ollama run <model>` with the prompt piped via stdin.
    Raises a RuntimeError with guidance if the daemon isn’t reachable.
    """
    prompt = (
        "Please provide a 2–4 sentence summary of the following text:\n\n"
        f"{chunk}\n\nSummary:"
    )
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True
    )
    if proc.returncode != 0:
        err = proc.stderr.strip()
        if "could not connect to ollama app" in err:
            print()
            print("❗️  Could not connect to the Ollama daemon.")
            print("   • Make sure you have Ollama installed and running:")
            print("       ollama serve")
            print("   • If you haven’t pulled the model yet, it will be pulled automatically.")
            print()
            sys.exit(1)
        raise RuntimeError(f"Ollama error: {err}")
    return proc.stdout.strip()

async def fetch_markdown(url: str) -> str:
    browser_cfg = BrowserConfig(headless=True)
    crawl_cfg   = CrawlerRunConfig()
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        res = await crawler.arun(url=url, config=crawl_cfg)
        if not res.success:
            raise RuntimeError(f"Failed to crawl {url}: {res.error_message}")
        return res.markdown

def chunk_text(text: str) -> list[str]:
    chunks = []
    start  = 0
    length = len(text)
    while start < length:
        end = min(start + MAX_CHARS, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - OVERLAP
    return chunks

def chunk_markdown(md: str) -> list[str]:
    header_pat = re.compile(r'^(# .+|## .+)$', re.MULTILINE)
    bounds = [m.start() for m in header_pat.finditer(md)] + [len(md)]
    subchunks = []
    for i in range(len(bounds) - 1):
        piece = md[bounds[i]:bounds[i+1]].strip()
        if piece:
            subchunks.append(piece)
    return subchunks

async def main():
    # 1) make sure model is available locally
    ensure_model(MODEL)

    # 2) fetch & chunk
    md = await fetch_markdown(URL)
    top_chunks = chunk_text(md)

    # 3) decide output folder
    if SUBFOLDER:
        folder = SUBFOLDER
    else:
        base = os.path.splitext(os.path.basename(urlparse(URL).path))[0] or "chunks"
        folder = f"{base}_chunks"
    out_dir = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)

    # 4) summarize & write
    counter = 1
    for big in top_chunks:
        for sub in chunk_markdown(big):
            summary = summarize_chunk(sub, MODEL)
            fn      = f"chunk_{counter:03}.md"
            path    = os.path.join(out_dir, fn)
            with open(path, "w", encoding="utf-8") as f:
                f.write("Summary:\n")
                f.write(summary + "\n\n")
                f.write(sub)
            counter += 1

    print(f"✅ Saved {counter-1} chunk files into {out_dir}")

if __name__ == "__main__":
    asyncio.run(main())
