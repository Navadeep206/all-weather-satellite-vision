import sys
import argparse
import requests
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"

def extract_uuid(text: str) -> Optional[str]:
    """Helper to extract a standard UUID from a string."""
    match = re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", text)
    return match.group(0) if match else None

def normalize_scene(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes a STAC item response into the project's internal scene representation."""
    properties = item.get("properties", {})
    item_id = item.get("id", "")
    
    # Determine product name and product ID (UUID)
    uuid = extract_uuid(item_id)
    if uuid:
        product_id = uuid
        product_name = properties.get("title") or item_id
    else:
        product_name = item_id
        # Search properties for OData UUID
        product_id = None
        for k, v in properties.items():
            if isinstance(v, str):
                extracted = extract_uuid(v)
                if extracted:
                    product_id = extracted
                    break
        
        # Search assets hrefs
        if not product_id:
            for asset in item.get("assets", {}).values():
                href = asset.get("href", "")
                extracted = extract_uuid(href)
                if extracted:
                    product_id = extracted
                    break
                    
        # Fallback
        if not product_id:
            product_id = item_id

    # Parse sensing datetime
    datetime_str = properties.get("datetime")
    
    # Parse tile ID (MGRS format) if available
    tile_id = None
    if "mgrs:utm_zone" in properties:
        tile_id = f"{properties['mgrs:utm_zone']}{properties.get('mgrs:latitude_band', '')}{properties.get('mgrs:grid_square', '')}"
    elif "s2:mgrs_tile" in properties:
        tile_id = properties["s2:mgrs_tile"]
        
    cloud_cover = properties.get("eo:cloud_cover", 100.0)
    platform = properties.get("platform", "Sentinel-2")
    
    # Get download URL if available in assets
    download_url = None
    if "download" in item.get("assets", {}):
        download_url = item["assets"]["download"].get("href")
    elif "data" in item.get("assets", {}):
        download_url = item["assets"]["data"].get("href")
    
    # Generate project scene ID, e.g. S2_L2A_<date>_T<tile>
    date_part = datetime_str[:10].replace("-", "") if datetime_str else "00000000"
    tile_part = tile_id if tile_id else "UNKNOWN"
    scene_id = f"S2_L2A_{date_part}_{tile_part}_{product_id[:8]}"
    
    return {
        "scene_id": scene_id,
        "product_id": product_id,
        "product_name": product_name,
        "collection": "sentinel-2-l2a",
        "processing_level": "L2A",
        "platform": platform,
        "sensing_datetime": datetime_str,
        "tile_id": tile_id,
        "cloud_cover": float(cloud_cover),
        "footprint": item.get("geometry"),
        "download_url": download_url,
        "source": "CDSE",
        "search_timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "discovered"
    }

def search_scenes(
    start_date: str,
    end_date: str,
    bbox: List[float],
    max_cloud_cover: float = 10.0,
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """Queries CDSE STAC API and filters results based on spatial/temporal/cloud limits.
    
    Args:
        start_date (str): Start date string YYYY-MM-DD.
        end_date (str): End date string YYYY-MM-DD.
        bbox (list of float): Bounding box coordinates [min_lon, min_lat, max_lon, max_lat].
        max_cloud_cover (float): Maximum allowed cloud cover percentage.
        max_results (int): Maximum number of normalized scenes to return.
        
    Returns:
        list of dict: List of normalized scene metadata records.
    """
    logger.info(f"Searching for Sentinel-2 L2A scenes from {start_date} to {end_date} in bbox {bbox}...")
    
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": 100  # Pull a large batch so we can filter cloud cover client-side
    }
    
    try:
        response = requests.post(STAC_SEARCH_URL, json=payload, timeout=30)
        response.raise_for_status()
        features = response.json().get("features", [])
        logger.info(f"Discovered {len(features)} total products from API query.")
    except Exception as e:
        logger.error(f"Failed to query CDSE STAC API: {e}")
        return []
        
    normalized_scenes = []
    for feat in features:
        # Check cloud cover client-side
        props = feat.get("properties", {})
        cloud_val = props.get("eo:cloud_cover", 100.0)
        
        if cloud_val <= max_cloud_cover:
            norm = normalize_scene(feat)
            normalized_scenes.append(norm)
            if len(normalized_scenes) >= max_results:
                break
                
    logger.info(f"Found {len(normalized_scenes)} scenes satisfying cloud cover limit of {max_cloud_cover}%.")
    return normalized_scenes

def main():
    parser = argparse.ArgumentParser(description="Search Sentinel-2 L2A scenes via CDSE STAC API")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="Bounding box coordinates")
    parser.add_argument("--max-cloud-cover", type=float, default=10.0, help="Max cloud cover percentage (default: 10.0)")
    parser.add_argument("--max-results", type=int, default=5, help="Max number of results to display/return (default: 5)")
    
    args = parser.parse_args()
    
    scenes = search_scenes(
        start_date=args.start_date,
        end_date=args.end_date,
        bbox=args.bbox,
        max_cloud_cover=args.max_cloud_cover,
        max_results=args.max_results
    )
    
    if not scenes:
        print("\nNo matching scenes found.")
        sys.exit(0)
        
    print("\n" + "=" * 105)
    print(f"{'Scene ID':<45} | {'Date':<20} | {'Tile':<8} | {'Cloud %':<8} | {'Platform':<12} | {'Level':<5}")
    print("=" * 105)
    for s in scenes:
        date_str = s["sensing_datetime"][:16] if s["sensing_datetime"] else "N/A"
        tile = s["tile_id"] if s["tile_id"] else "N/A"
        cloud = f"{s['cloud_cover']:.1f}%"
        print(f"{s['scene_id']:<45} | {date_str:<20} | {tile:<8} | {cloud:<8} | {s['platform']:<12} | {s['processing_level']:<5}")
    print("=" * 105 + "\n")

if __name__ == "__main__":
    main()
