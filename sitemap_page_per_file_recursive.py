import os
import json
import re
import sys
import subprocess
import asyncio
import requests
import html2text
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

# ──────────── CONFIGURATION ────────────
# Can be a sitemap.xml or a single .md/.txt URL
# URL        = "https://example.com/sitemap.xml"
#URL        = "https://rpstrength.com/sitemap_blogs_1.xml"
URL        = "https://faroutride.com/sitemap_index.xml"
OUTPUT_DIR = "out_posts"
MODEL      = "gemma3:1b-it-qat"
# ────────────────────────────────────────

# A realistic User-Agent string so servers don’t 403 us
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}

def fetch_sitemap_urls(sitemap_url: str,
                       visited: set[str] = None
                      ) -> list[str]:
    """
    Recursively fetch all <loc> URLs from a sitemap or sitemap-index,
    using a browser-like User-Agent to avoid 403s.
    """
    if visited is None:
        visited = set()
    if sitemap_url in visited:
        return []
    visited.add(sitemap_url)

    # use a Session with our headers
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    resp = session.get(sitemap_url, timeout=10)
    resp.raise_for_status()
    data = resp.content

    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        # fallback: regex scrape of all <loc>…</loc>
        text = data.decode("utf-8", errors="ignore")
        return re.findall(r"<loc>\s*(.*?)\s*</loc>", text)

    tag = root.tag.split("}")[-1].lower()
    urls: list[str] = []

    if tag == "sitemapindex":
        for sm in root.findall(".//{*}sitemap"):
            loc = sm.find("{*}loc")
            if loc is not None and loc.text:
                child = loc.text.strip()
                urls += fetch_sitemap_urls(child, visited)
    elif tag == "urlset":
        for urln in root.findall(".//{*}url"):
            loc = urln.find("{*}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
    else:
        # unknown variant: grab every <loc>
        for loc in root.findall(".//{*}loc"):
            if loc.text:
                urls.append(loc.text.strip())

    return urls

def ensure_model(model: str):
    """
    Ensure that `model` is present locally.
    1) Try `ollama list models --json`
    2) Fallback to parsing `ollama list models`
    3) If *neither* yields any names, warn & skip auto-pull.
    4) Otherwise, pull only if missing.
    """
    names = []
    # 1) JSON listing
    try:
        proc = subprocess.run(
            ["ollama", "list", "models", "--json"],
            capture_output=True, text=True
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            # JSON is usually a list of dicts with "name" keys
            names = [m.get("name") for m in data if isinstance(m, dict) and m.get("name")]
    except Exception:
        pass

    # 2) Plain-text fallback
    if not names:
        proc = subprocess.run(
            ["ollama", "list", "models"],
            capture_output=True, text=True
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                # skip headers or empty lines
                if not line or line.lower().startswith("name"):
                    continue
                # first whitespace-delimited token is the model name
                parts = line.split()
                if parts:
                    names.append(parts[0])

    # 3) If we still got nothing, bail out of auto-pull
    if not names:
        print(f"⚠️  Could not detect any local Ollama models (parsed 0 names).")
        print(f"   Please ensure you have already run:\n     ollama pull {model}\n")
        return

    # 4) If your model isn’t in the list, pull it
    if model not in names:
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
        "Please provide a formal 2–4 sentence summary of the main portion of the following text, please focus on the main portion of the text and ignore the header sorts of stuff like menue and search:\n\n"
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
    """
    1) Try dynamic scrape via Crawl4AI/Playwright, completely silencing its logs.
    2) On any error, silently fall back to requests+html2text.
    3) Only truly unhandled errors (like network down) will raise.
    """
    browser_cfg = BrowserConfig(headless=True)
    crawl_cfg   = CrawlerRunConfig(
        page_timeout=30_000,
        wait_for="css:body"
    )

    # 1) Dynamic scrape (silent)
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                async with AsyncWebCrawler(config=browser_cfg) as crawler:
                    res = await crawler.arun(url=url, config=crawl_cfg)
        if res.success and res.markdown:
            return res.markdown
    except Exception:
        # fully swallow any Crawl4AI/playwright errors
        pass

    # 2) Static fallback
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
    resp.raise_for_status()
    html = resp.text

    conv = html2text.HTML2Text()
    conv.ignore_links = False
    conv.body_width   = 0
    return conv.handle(html)

async def main():
    ensure_model(MODEL)

    # build list of pages
    if URL.lower().endswith(".xml") or "sitemap" in URL.lower():
        targets = fetch_sitemap_urls(URL)
        print(f"🔍 Discovered {len(targets)} page URLs via sitemap recursion.")
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
        parsed = urlparse(target)
        # strip trailing slash so basename() isn’t empty
        clean_path = parsed.path.rstrip("/")  
        base = os.path.basename(clean_path)
        slug = base or "index"
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
