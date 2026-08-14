import os
import time
import requests
from typing import Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class CDSEClient:
    """Client for authenticating and downloading data from the Copernicus Data Space Ecosystem (CDSE)."""
    
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    DOWNLOAD_BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """Initializes the CDSE Client with credentials.
        
        If credentials are not provided, it attempts to load them from environment variables.
        """
        self.username = username or os.environ.get("CDSE_USERNAME")
        self.password = password or os.environ.get("CDSE_PASSWORD")
        self._access_token: Optional[str] = None
        self._token_expiry_time: float = 0.0

    def has_credentials(self) -> bool:
        """Checks if both username and password are provided."""
        return bool(self.username and self.password)

    def get_access_token(self) -> str:
        """Obtains an access token from Keycloak, using the cached token if it's still valid.
        
        Returns:
            str: The active access token.
            
        Raises:
            ValueError: If credentials are missing or authentication fails.
        """
        if not self.has_credentials():
            raise ValueError(
                "CDSE credentials missing. Please set CDSE_USERNAME and CDSE_PASSWORD env variables."
            )
            
        # If token is still valid (with a 60-second safety buffer), return it
        if self._access_token and time.time() < self._token_expiry_time - 60:
            return self._access_token

        logger.info("Requesting new CDSE access token...")
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        
        try:
            response = requests.post(self.TOKEN_URL, data=data, timeout=30)
            response.raise_for_status()
            res_json = response.json()
            
            self._access_token = res_json["access_token"]
            expires_in = res_json.get("expires_in", 900)  # default to 15 minutes
            self._token_expiry_time = time.time() + expires_in
            
            logger.info("CDSE access token acquired successfully.")
            return self._access_token
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                logger.error("Authentication failed: Invalid CDSE username or password.")
                raise ValueError("Authentication failed: Invalid CDSE username or password.") from e
            logger.error(f"HTTP error during authentication: {e}")
            raise RuntimeError(f"Authentication failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during CDSE authentication: {e}")
            raise RuntimeError(f"Authentication failed: {e}") from e

    def download_product(
        self,
        product_id: str,
        output_path: Path,
        max_retries: int = 5,
        backoff_factor: float = 2.0
    ) -> Path:
        """Downloads a product from CDSE by its product ID.
        
        Args:
            product_id (str): The unique CDSE product UUID.
            output_path (Path): File path where the downloaded zip will be saved.
            max_retries (int): Maximum number of retry attempts for network failures.
            backoff_factor (float): Multiplier for exponential backoff sleep time.
            
        Returns:
            Path: The path to the downloaded file.
            
        Raises:
            RuntimeError: If download fails after max retries.
        """
        token = self.get_access_token()
        download_url = f"{self.DOWNLOAD_BASE_URL}({product_id})/$value"
        headers = {"Authorization": f"Bearer {token}"}
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = output_path.with_suffix(".tmp")
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.info(f"Downloading product {product_id} (Attempt {retry_count + 1}/{max_retries})...")
                # Using stream=True to handle large files efficiently
                with requests.get(download_url, headers=headers, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    
                    total_size = int(r.headers.get("content-length", 0))
                    chunk_size = 1024 * 1024  # 1MB chunks
                    
                    with open(temp_output_path, "wb") as f, tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Product {product_id[:8]}",
                        leave=False
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                                
                # Rename temp file to actual file upon completion
                if temp_output_path.exists():
                    temp_output_path.rename(output_path)
                logger.info(f"Product {product_id} downloaded successfully to {output_path}")
                return output_path
                
            except (requests.RequestException, IOError) as e:
                retry_count += 1
                logger.warning(f"Download failed on attempt {retry_count}: {e}")
                
                if temp_output_path.exists():
                    try:
                        temp_output_path.unlink()
                    except Exception:
                        pass
                        
                if retry_count >= max_retries:
                    logger.error(f"Failed to download product {product_id} after {max_retries} attempts.")
                    raise RuntimeError(f"Failed to download product {product_id}: {e}") from e
                    
                sleep_time = backoff_factor ** retry_count
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                
        raise RuntimeError(f"Failed to download product {product_id} due to unknown error.")
