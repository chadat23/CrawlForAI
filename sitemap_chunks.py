import os
import re
import sys
import subprocess
import asyncio
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

# ──────────── CONFIGURATION ────────────
# URL of the sitemap.xml or single Markdown page
#URL        = "https://example.com/sitemap.xml"
URL        = "https://rpstrength.com/sitemap_blogs_1.xml"
# Base directory where chunk folders will be created
OUTPUT_DIR = "out_chunks"
# Name of the subfolder under OUTPUT_DIR (None to auto-derive)
SUBFOLDER  = None
# Ollama model to use for summarization
MODEL      = "gemma3:1b-it-qat"
# Chunking parameters
MAX_CHARS  = 2000   # max characters per chunk
OVERLAP    = 200    # overlapping characters between chunks
# ────────────────────────────────────────

def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    """
    Download and parse sitemap.xml, return all <loc> URLs.
    Falls back to regex if XML parsing fails.
    """
    resp = requests.get(sitemap_url)
    resp.raise_for_status()
    data = resp.content
    try:
        root = ET.fromstring(data)
        # sitemap namespace
        ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        #ns = {"s": "https://rpstrength.com/sitemap_blogs_1.xml"}
        locs = root.findall(".//s:loc", ns)
        if not locs:
            locs = root.findall(".//loc")
        return [loc.text.strip() for loc in locs if loc.text]
    except ET.ParseError:
        # fallback regex
        text = data.decode("utf-8", errors="ignore")
        return re.findall(r"<loc>(.*?)</loc>", text)

def ensure_model(model: str):
    """
    Ensure the Ollama model is pulled locally.
    """
    lst = subprocess.run(
        ["ollama", "list", "models"],
        capture_output=True, text=True
    )
    if lst.returncode != 0:
        print("❗️ Cannot run `ollama list models`. Is Ollama installed?")
        sys.exit(1)
    if model not in lst.stdout:
        print(f"⬇️ Pulling model {model}…")
        pull = subprocess.run(["ollama", "pull", model])
        if pull.returncode != 0:
            print(f"❗️ Failed to pull {model}")
            sys.exit(1)

def summarize_chunk(chunk: str, model: str) -> str:
    """
    Run `ollama run <model>` with prompt on stdin, decode as UTF-8.
    """
    prompt = (
        "Please provide a 2–4 sentence summary of the following text:\n\n"
        f"{chunk}\n\nSummary:"
    )
    proc = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out_bytes, err_bytes = proc.communicate(prompt.encode("utf-8"))
    out = out_bytes.decode("utf-8", errors="replace")
    err = err_bytes.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        if "could not connect to ollama app" in err:
            print("\n❗️ Could not connect to the Ollama daemon.")
            print("   • Start it with: ollama serve\n")
            sys.exit(1)
        raise RuntimeError(f"Ollama error: {err.strip()}")
    return out.strip()

async def fetch_markdown(url: str) -> str:
    """
    Crawl the URL via Crawl4AI, return Markdown text.
    """
    browser_cfg = BrowserConfig(headless=True)
    crawl_cfg   = CrawlerRunConfig()
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        res = await crawler.arun(url=url, config=crawl_cfg)
        if not res.success:
            raise RuntimeError(f"Failed to crawl {url}: {res.error_message}")
        return res.markdown

def chunk_text(text: str) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.
    """
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
    """
    Split a markdown chunk further at H1/H2 headers.
    """
    header_pat = re.compile(r'^(# .+|## .+)$', re.MULTILINE)
    bounds = [m.start() for m in header_pat.finditer(md)] + [len(md)]
    subs = []
    for i in range(len(bounds) - 1):
        piece = md[bounds[i]:bounds[i+1]].strip()
        if piece:
            subs.append(piece)
    return subs


def chunk_by_header(markdown: str) -> list[str]:
    """
    Split the entire MD into header‐led sections
    (each starts with a `# ` or `## ` header).
    """
    header_pat = re.compile(r'^(# .+|## .+)$', re.MULTILINE)
    bounds = [m.start() for m in header_pat.finditer(markdown)] + [len(markdown)]
    sections = []
    for i in range(len(bounds)-1):
        sec = markdown[bounds[i]:bounds[i+1]].strip()
        if sec:
            sections.append(sec)
    return sections

def chunk_by_paragraphs(section: str,
                        max_chars: int = 2000,
                        overlap_paras: int = 1
                       ) -> list[str]:
    """
    Take a header‐section and split into chunks by whole paragraphs,
    each ≤ max_chars. We optionally repeat the last N paragraphs
    of a chunk at the start of the next to give overlap context.
    """
    paras = re.split(r'\n\s*\n', section.strip())
    chunks = []
    curr = []
    curr_len = 0

    for p in paras:
        plen = len(p) + 2  # account for the blank line
        if curr_len + plen > max_chars and curr:
            # flush current
            chunks.append("\n\n".join(curr).strip())
            # keep the last overlap_paras for context
            curr = curr[-overlap_paras:]
            curr_len = sum(len(x)+2 for x in curr)
        curr.append(p)
        curr_len += plen

    if curr:
        chunks.append("\n\n".join(curr).strip())
    return chunks

async def main():
    # 1) Ensure model is available
    ensure_model(MODEL)

    # 2) Build list of target URLs
    if URL.lower().endswith(".xml") or "sitemap" in URL.lower():
        targets = fetch_sitemap_urls(URL)
        print(f"🔍 Found {len(targets)} URLs in sitemap.")
    else:
        targets = [URL]

    counter = 1
    for target in targets:
        print(f"\n▶️ Processing {target}")
        try:
            md = await fetch_markdown(target)
        except Exception as e:
            # <-- catch *any* failure in crawling
            print(f"⚠️  Skipping {target} due to error:\n    {e}")
            continue

        # chunk + summarize as before
        sections = chunk_by_header(md)
        for sec in sections:
            for sub in chunk_by_paragraphs(sec, MAX_CHARS, overlap_paras=1):
                summary = summarize_chunk(sub, MODEL)
                fn      = f"chunk_{counter:05}.md"
                out_dir = os.path.join(
                    OUTPUT_DIR,
                    SUBFOLDER or f"{os.path.splitext(os.path.basename(urlparse(target).path))[0] or 'page'}_chunks"
                )
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, fn)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("Summary:\n")
                    f.write(summary + "\n\n")
                    f.write(sub)
                counter += 1

    print(f"\n✅ Saved {counter-1} chunk files into {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
