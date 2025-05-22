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
# Can be a sitemap.xml or a single .md/.txt URL
# URL        = "https://example.com/sitemap.xml"
URL        = "https://rpstrength.com/sitemap_blogs_1.xml"
OUTPUT_DIR = "out_posts"
MODEL      = "gemma3:1b-it-qat"
# ────────────────────────────────────────

def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    """Download & parse <loc> entries from sitemap.xml."""
    resp = requests.get(sitemap_url)
    resp.raise_for_status()
    data = resp.content
    try:
        root = ET.fromstring(data)
        ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locs = root.findall(".//s:loc", ns) or root.findall(".//loc")
        return [loc.text.strip() for loc in locs if loc.text]
    except ET.ParseError:
        text = data.decode("utf-8", errors="ignore")
        return re.findall(r"<loc>(.*?)</loc>", text)

def ensure_model(model: str):
    """Pull the Ollama model if it isn’t already present."""
    lst = subprocess.run(
        ["ollama", "list", "models"], capture_output=True, text=True
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

def summarize_chunk(text: str, model: str) -> str:
    """
    Summarize *all* of `text` in 2–4 sentences, feeding prompt via stdin.
    """
    prompt = (
        "Please provide a 2–4 sentence summary of the following text:\n\n"
        f"{text}\n\nSummary:"
    )
    proc = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    outb, errb = proc.communicate(prompt.encode("utf-8"))
    out = outb.decode("utf-8", errors="replace").strip()
    err = errb.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        if "could not connect to ollama app" in err.lower():
            print("\n❗️ Could not connect to Ollama daemon. Run `ollama serve`.")
            sys.exit(1)
        raise RuntimeError(f"Ollama error: {err.strip()}")
    return out

async def fetch_markdown(url: str) -> str:
    """Use Crawl4AI to scrape the page and return its Markdown."""
    browser_cfg = BrowserConfig(headless=True)
    crawl_cfg   = CrawlerRunConfig()
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        res = await crawler.arun(url=url, config=crawl_cfg)
        if not res.success:
            raise RuntimeError(f"Failed to crawl {url}: {res.error_message}")
        return res.markdown

async def main():
    ensure_model(MODEL)

    # build list of pages
    if URL.lower().endswith(".xml") or "sitemap" in URL.lower():
        targets = fetch_sitemap_urls(URL)
        print(f"🔍 Found {len(targets)} URLs in sitemap.")
    else:
        targets = [URL]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    count = 0

    for target in targets:
        print(f"\n▶️  Processing {target}")
        try:
            md = await fetch_markdown(target)
        except Exception as e:
            print(f"⚠️  Skipping {target} due to: {e}")
            continue

        summary = summarize_chunk(md, MODEL)

        # derive a safe filename from the URL path
        slug = os.path.splitext(os.path.basename(urlparse(target).path))[0] or "index"
        filename = f"{slug}.md"
        out_path = os.path.join(OUTPUT_DIR, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("Summary:\n")
            f.write(summary + "\n\n")
            f.write(md)

        count += 1
        print(f"✅ Wrote {out_path}")

    print(f"\n🎉 Completed. Wrote {count} files into {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
