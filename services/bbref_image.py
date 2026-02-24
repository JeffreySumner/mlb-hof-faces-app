"""Baseball Reference image retrieval service with caching."""

from io import BytesIO
from pathlib import Path
import time

from bs4 import BeautifulSoup
from PIL import Image
import requests

try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except Exception:
    HAS_CURL_CFFI = False

from config import IMAGE_CACHE_DIR


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


def _direct_headshot_urls(bbref_id: str) -> list[str]:
    first_letter = bbref_id[0]
    return [
        f"https://cdn.ssref.net/req/202411010/images/headshots/{first_letter}/{bbref_id}.jpg",
        f"https://cdn.ssref.net/req/202408150/images/headshots/{first_letter}/{bbref_id}.jpg",
        f"https://www.baseball-reference.com/req/202411010/images/headshots/{first_letter}/{bbref_id}.jpg",
        f"https://www.baseball-reference.com/req/202408150/images/headshots/{first_letter}/{bbref_id}.jpg",
        f"https://cdn.ssref.net/req/1/images/headshots/{first_letter}/{bbref_id}.jpg",
    ]


def _player_url(bbref_id: str) -> str:
    bbref_id = bbref_id.strip().lower()
    first_letter = bbref_id[0]
    return f"https://www.baseball-reference.com/players/{first_letter}/{bbref_id}.shtml"


def _extract_headshot_url(page_html: bytes) -> str:
    soup = BeautifulSoup(page_html, "html.parser")

    img_tag = soup.find("img", {"itemprop": "image"})
    if not img_tag:
        imgs = soup.find_all("img")
        if len(imgs) < 2:
            raise ValueError("Could not find player image tag on page.")
        img_tag = imgs[1]

    img_url = (img_tag.get("src") or "").strip()
    if not img_url:
        raise ValueError("Image URL missing in player page.")
    if img_url.startswith("//"):
        img_url = f"https:{img_url}"
    elif img_url.startswith("/"):
        img_url = f"https://www.baseball-reference.com{img_url}"
    return img_url


def _is_bad_image(img: Image.Image) -> bool:
    # Simple placeholder detection: nearly monochrome thumbnails
    sample = list(img.convert("RGB").getdata())[:200]
    return len(set(sample)) < 5


def _session():
    if HAS_CURL_CFFI:
        s = curl_requests.Session(impersonate="chrome")
        s.headers.update(HEADERS)
        return s
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _try_fetch_direct(session, bbref_id: str) -> Image.Image | None:
    for url in _direct_headshot_urls(bbref_id):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 1000:
                img = Image.open(BytesIO(resp.content))
                if not _is_bad_image(img):
                    return img
        except Exception:
            continue
    return None


def fetch_player_image(bbref_id: str, force_refresh: bool = False) -> Path:
    """Fetch a player's headshot image and return local cache path."""
    bbref_id = bbref_id.strip().lower()
    if not bbref_id:
        raise ValueError("bbref_id is required.")

    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGE_CACHE_DIR / f"{bbref_id}.jpg"
    if out_path.exists() and not force_refresh:
        return out_path

    # Be polite to Baseball Reference
    time.sleep(0.8)

    session = _session()

    # Try direct CDN headshot endpoints first (often avoids 403 page fetches)
    img = _try_fetch_direct(session, bbref_id)
    if img is None:
        page_resp = session.get(_player_url(bbref_id), timeout=20)
        page_resp.raise_for_status()

        img_url = _extract_headshot_url(page_resp.content)
        img_resp = session.get(img_url, timeout=20)
        img_resp.raise_for_status()
        img = Image.open(BytesIO(img_resp.content))

    if _is_bad_image(img):
        raise ValueError("Retrieved a placeholder/invalid image.")

    img.convert("RGB").save(out_path, format="JPEG")
    return out_path

