"""
Baseball Reference image scraper utilities.
Provides functions to fetch a player's headshot image and retrieve the player's full name.
"""

import time
import random
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image

# Try to import curl_cffi for better TLS fingerprinting (bypasses Cloudflare)
try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except ImportError:
    import requests
    HAS_CURL_CFFI = False

# Fallback to regular requests if curl_cffi not available
if not HAS_CURL_CFFI:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

def _parse_html(content):
    """Parse HTML content with BeautifulSoup, falling back to html.parser if lxml unavailable."""
    try:
        return BeautifulSoup(content, "lxml")
    except Exception:
        return BeautifulSoup(content, "html.parser")


# Common headers for fallback requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _create_session():
    """Create a session for making HTTP requests.
    
    Uses curl_cffi with Chrome impersonation if available (bypasses Cloudflare),
    otherwise falls back to regular requests with retry logic.
    """
    if HAS_CURL_CFFI:
        # curl_cffi with Chrome TLS fingerprint - bypasses Cloudflare
        return curl_requests.Session(impersonate='chrome')
    else:
        # Fallback to regular requests with retry logic
        session = requests.Session()
        session.headers.update(HEADERS)
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session


def _build_player_url(player_id: str) -> str:
    """Construct the Baseball‑Reference URL for a given player ID."""
    player_id = player_id.strip()
    first_letter = player_id[0].lower()
    return f"https://www.baseball-reference.com/players/{first_letter}/{player_id}.shtml"


def _get_direct_image_urls(player_id: str) -> list:
    """Generate list of potential direct image URLs to try."""
    player_id = player_id.strip()
    first_letter = player_id[0].lower()
    
    # Multiple URL patterns that Baseball Reference uses for headshots
    return [
        # CDN URLs (often less protected)
        f"https://cdn.ssref.net/req/202411010/images/headshots/{first_letter}/{player_id}.jpg",
        f"https://cdn.ssref.net/req/202408150/images/headshots/{first_letter}/{player_id}.jpg",
        f"https://cdn.ssref.net/req/202311010/images/headshots/{first_letter}/{player_id}.jpg",
        # Direct baseball-reference URLs with different date codes
        f"https://www.baseball-reference.com/req/202411010/images/headshots/{first_letter}/{player_id}.jpg",
        f"https://www.baseball-reference.com/req/202408150/images/headshots/{first_letter}/{player_id}.jpg",
        # Alternative path patterns
        f"https://cdn.ssref.net/req/1/images/headshots/{first_letter}/{player_id}.jpg",
        f"https://cdn.ssref.net/req/202411010/images/br/headshots/{first_letter}/{player_id}.jpg",
        # Stathead CDN (same parent company)
        f"https://stathead.com/req/202411010/images/headshots/{first_letter}/{player_id}.jpg",
    ]


def _is_placeholder_image(path: Path) -> bool:
    """Check if an image file is a placeholder (gray/single color).
    
    Returns True if the image has very few unique colors (likely placeholder).
    """
    try:
        with Image.open(path) as img:
            # Sample first 100 pixels
            pixels = list(img.convert('RGB').getdata())[:100]
            unique_colors = len(set(pixels))
            # Placeholder images typically have 1-3 unique colors
            return unique_colors < 5
    except Exception:
        return True  # If we can't read it, treat as placeholder


def _check_existing_images(player_id: str, validate: bool = True) -> Path | None:
    """Check if player image already exists in project data folders.
    
    Parameters
    ----------
    player_id : str
        The player's Baseball Reference ID.
    validate : bool
        If True, validates that the image is not a placeholder.
        Placeholder images will be deleted and None returned.
    """
    player_id = player_id.strip()
    
    # Possible locations for existing images
    possible_paths = [
        Path(__file__).parent.parent / "data" / "Hall_of_Fame_Eligible" / f"eligibility_{player_id}.jpg",
        Path(__file__).parent.parent / "data" / "Hall_of_Fame_Eligible" / f"N_{player_id}.jpg",
        Path(__file__).parent.parent / "data" / "Hall_of_Fame_Eligible" / f"Y_{player_id}.jpg",
        # Main rpy project data folder
        Path(__file__).parent.parent.parent / "data" / "MLB_Hall_Of_Fame_Project" / "Hall_of_Fame_Eligible" / f"N_{player_id}.jpg",
        Path(__file__).parent.parent.parent / "data" / "MLB_Hall_Of_Fame_Project" / "Hall_of_Fame_Eligible" / f"Y_{player_id}.jpg",
    ]
    
    for path in possible_paths:
        if path.exists():
            if validate and _is_placeholder_image(path):
                # Delete placeholder and continue searching/re-fetch
                try:
                    path.unlink()
                except Exception:
                    pass
                continue
            return path
    
    return None


def _fetch_image_direct(player_id: str, session = None) -> Image.Image:
    """Try to fetch player image directly from CDN URLs, bypassing the player page."""
    if session is None:
        session = _create_session()
    
    urls = _get_direct_image_urls(player_id)
    
    for url in urls:
        try:
            time.sleep(random.uniform(0.3, 0.8))
            resp = session.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 1000:  # Ensure it's actually an image
                return Image.open(BytesIO(resp.content))
        except Exception:
            continue
    
    return None


def _create_placeholder_image(player_id: str) -> Image.Image:
    """Create a placeholder image when player image cannot be fetched."""
    from PIL import ImageDraw, ImageFont
    
    # Create a gray placeholder image
    img = Image.new('RGB', (120, 180), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Add text
    text = player_id[:10]  # Truncate if too long
    try:
        # Try to use a basic font
        font = ImageFont.load_default()
    except Exception:
        font = None
    
    # Center the text
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = 60, 10
    
    x = (120 - text_width) // 2
    y = (180 - text_height) // 2
    
    draw.text((x, y), text, fill=(100, 100, 100), font=font)
    draw.text((x, y + 20), "No Image", fill=(100, 100, 100), font=font)
    
    return img


def scrape_player_image(player_id: str) -> Image.Image:
    """Download the first head‑shot image for a player.

    Parameters
    ----------
    player_id: str
        The Baseball‑Reference player identifier (e.g. "bondsba01").

    Returns
    -------
    PIL.Image.Image
        The player's head‑shot image.
    """
    player_id = player_id.strip()
    
    # Check if we already have this image locally
    existing_path = _check_existing_images(player_id)
    if existing_path:
        return Image.open(existing_path)
    
    session = _create_session()
    
    # Try direct CDN URLs first (faster and less likely to be blocked)
    direct_image = _fetch_image_direct(player_id, session)
    if direct_image:
        return direct_image
    
    # If direct fetch failed, try the player page as last resort
    url = _build_player_url(player_id)
    time.sleep(random.uniform(1.5, 3.0))
    
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
    except Exception:
        # All methods failed - return a placeholder image
        return _create_placeholder_image(player_id)

    soup = _parse_html(response.content)

    # Prefer the semantic image tag used by Baseball‑Reference
    img_tag = soup.find("img", {"itemprop": "image"})
    if not img_tag:
        # Fallback: use the second <img> on the page (first is usually a logo)
        img_tags = soup.find_all("img")
        if len(img_tags) < 2:
            raise ValueError(f"No player image found for {player_id}")
        img_tag = img_tags[1]

    img_url = img_tag.get("src")
    if not img_url:
        raise ValueError(f"Image URL missing for player {player_id}")

    # Some URLs are relative – make them absolute
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif img_url.startswith("/"):
        img_url = f"https://www.baseball-reference.com{img_url}"

    try:
        time.sleep(random.uniform(0.5, 1.0))
        img_resp = session.get(img_url, timeout=15)
        img_resp.raise_for_status()
        return Image.open(BytesIO(img_resp.content))
    except Exception as e:
        raise ValueError(f"Failed to download player image: {e}") from e


def get_player_name(player_id: str) -> str:
    """Retrieve a player's full name from Baseball‑Reference.

    Parameters
    ----------
    player_id: str
        The Baseball‑Reference player identifier.

    Returns
    -------
    str
        The player's displayed name, or the raw ID if it cannot be determined.
    """
    player_id = player_id.strip()
    url = _build_player_url(player_id)

    session = _create_session()
    
    try:
        time.sleep(random.uniform(1.0, 2.0))
        response = session.get(url, timeout=15)
        response.raise_for_status()
    except Exception:
        return player_id

    soup = _parse_html(response.content)
    h1 = soup.find("h1")
    if h1:
        name = h1.get_text().strip()
        # Remove any "HOF" badge that might be appended
        name = name.replace("HOF", "").strip()
        return name
    return player_id
def download_player_image(player_id: str, dest_dir: Path = Path(__file__).parent.parent / "data" / "Hall_of_Fame_Eligible") -> Path:
    """Download a player's headshot image and save it locally.

    The image is saved as ``eligibility_{player_id}.jpg`` inside the
    ``Hall_of_Fame_Eligible`` directory. If the file already exists the
    download is skipped.
    """
    player_id = player_id.strip()
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / f"eligibility_{player_id}.jpg"
    if file_path.exists():
        return file_path
    
    # Check if we already have this image in another location
    existing_path = _check_existing_images(player_id)
    if existing_path:
        # Copy to expected location
        import shutil
        shutil.copy(existing_path, file_path)
        return file_path
    
    session = _create_session()
    
    # Try direct CDN URLs first (faster and less likely to be blocked)
    urls = _get_direct_image_urls(player_id)
    
    for url in urls:
        try:
            time.sleep(random.uniform(0.3, 0.8))
            resp = session.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(file_path, "wb") as f:
                    f.write(resp.content)
                return file_path
        except Exception:
            continue
    
    # If direct fetch failed, try the player page as last resort
    url = _build_player_url(player_id)
    time.sleep(random.uniform(1.5, 3.0))
    
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
    except Exception:
        # All methods failed - create and save a placeholder image
        placeholder = _create_placeholder_image(player_id)
        placeholder.save(file_path, "JPEG")
        return file_path
    
    soup = BeautifulSoup(response.content, "lxml")
    img_tag = soup.find("img", {"itemprop": "image"})
    if not img_tag:
        img_tags = soup.find_all("img")
        if len(img_tags) < 2:
            # No image found - create placeholder
            placeholder = _create_placeholder_image(player_id)
            placeholder.save(file_path, "JPEG")
            return file_path
        img_tag = img_tags[1]
    img_url = img_tag.get("src")
    if not img_url:
        placeholder = _create_placeholder_image(player_id)
        placeholder.save(file_path, "JPEG")
        return file_path
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif img_url.startswith("/"):
        img_url = f"https://www.baseball-reference.com{img_url}"
    try:
        time.sleep(random.uniform(0.5, 1.0))
        img_resp = session.get(img_url, timeout=15)
        img_resp.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(img_resp.content)
        return file_path
    except Exception:
        # Failed to download - create placeholder
        placeholder = _create_placeholder_image(player_id)
        placeholder.save(file_path, "JPEG")
        return file_path
def scrape_bbref_images(baseball_reference_id: str) -> Path:
    """Public wrapper matching the R function name.

    Downloads the player's headshot image into the ``Hall_of_Fame_Eligible``
    directory (if not already present) and returns the absolute path to the
    saved file.
    """
    return download_player_image(baseball_reference_id)
