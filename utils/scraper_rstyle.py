"""
R-style Baseball Reference scraper.
Mimics the R rvest::session() approach for fetching player images.
"""

import time
import random
from pathlib import Path
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PIL import Image


def scrape_bbref_image_rstyle(
    baseball_reference_id: str,
    dest_dir: Path = None,
    save_image: bool = False
) -> dict:
    """
    Scrape player image from Baseball Reference using R-style approach.
    
    Mirrors the R function:
    ```r
    scrape_bbref_images <- function(baseball_reference_id){
      first_letter_first_name <- substr(baseball_reference_id,1,1)
      url <- glue::glue("https://www.baseball-reference.com/players/{first_letter_first_name}/{baseball_reference_id}.shtml")
      webpage <- session(url)
      link_titles <- webpage %>% html_nodes("img")
      img_url_first <- link_titles[2] %>% html_attr("src")
    }
    ```
    
    Parameters
    ----------
    baseball_reference_id : str
        The player ID from Baseball Reference (e.g., "ruthba01")
    dest_dir : Path, optional
        Directory to save the image
    save_image : bool
        Whether to download and save the image
        
    Returns
    -------
    dict
        Dictionary with 'player_id', 'url', 'img_url', 'status', and optionally 'image' or 'file_path'
    """
    baseball_reference_id = baseball_reference_id.strip()
    first_letter = baseball_reference_id[0].lower()
    
    url = f"https://www.baseball-reference.com/players/{first_letter}/{baseball_reference_id}.shtml"
    
    result = {
        'player_id': baseball_reference_id,
        'url': url,
        'img_url': None,
        'status': None,
        'error': None
    }
    
    # Create session mimicking R's rvest::session()
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    print(f"Fetching: {baseball_reference_id}")
    
    # R uses Sys.sleep(5) - we'll use a random delay
    time.sleep(random.uniform(3, 5))
    
    try:
        response = session.get(url, timeout=15)
        result['status'] = response.status_code
        
        if response.status_code == 403:
            result['error'] = "403 Forbidden - Baseball Reference is blocking automated requests"
            print(f"  Status: 403 Forbidden")
            return result
            
        response.raise_for_status()
        
    except requests.RequestException as e:
        result['error'] = str(e)
        print(f"  Error: {e}")
        return result
    
    # Parse HTML like R's html_nodes("img")
    soup = BeautifulSoup(response.content, "lxml")
    
    # Get all img tags (like R's webpage %>% html_nodes("img"))
    img_tags = soup.find_all("img")
    
    # R uses link_titles[2] (1-indexed), so in Python it's index 1 (0-indexed)
    if len(img_tags) >= 2:
        img_url_first = img_tags[1].get("src")  # R's html_attr("src")
        
        # Make URL absolute if relative
        if img_url_first:
            if img_url_first.startswith("//"):
                img_url_first = "https:" + img_url_first
            elif img_url_first.startswith("/"):
                img_url_first = f"https://www.baseball-reference.com{img_url_first}"
        
        result['img_url'] = img_url_first
        print(f"  Image URL: {img_url_first}")
        
        # Download image if requested
        if save_image and img_url_first and dest_dir:
            try:
                time.sleep(random.uniform(0.5, 1.5))
                img_resp = session.get(img_url_first, timeout=15)
                img_resp.raise_for_status()
                
                dest_dir = Path(dest_dir)
                dest_dir.mkdir(parents=True, exist_ok=True)
                file_path = dest_dir / f"{baseball_reference_id}.jpg"
                
                with open(file_path, "wb") as f:
                    f.write(img_resp.content)
                
                result['file_path'] = str(file_path)
                result['image'] = Image.open(BytesIO(img_resp.content))
                print(f"  Saved to: {file_path}")
                
            except Exception as e:
                result['error'] = f"Failed to download image: {e}"
                print(f"  Download error: {e}")
    else:
        result['error'] = "Could not find player image (less than 2 img tags)"
        print(f"  Error: Not enough img tags found")
    
    return result


def test_scraper():
    """Test the R-style scraper with a few players."""
    test_players = ["ruthba01", "aaronha01", "bondsba01"]
    
    print("=" * 60)
    print("Testing R-style Baseball Reference Scraper")
    print("=" * 60)
    
    for player_id in test_players:
        print(f"\n--- {player_id} ---")
        result = scrape_bbref_image_rstyle(player_id)
        print(f"Result: {result}")
        print()


if __name__ == "__main__":
    test_scraper()
