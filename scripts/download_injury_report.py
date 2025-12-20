"""
RotoWire College Basketball Injury Report Downloader
This script downloads the latest injury report from RotoWire and saves it to your project directory.

Features:
- Verifies and creates target directory if needed
- Automatic file overwriting
- Dynamic date-based naming (YYYYMMDD format)
- Comprehensive error handling and logging
- Can be scheduled via Windows Task Scheduler
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error
from bs4 import BeautifulSoup
import csv
import json

# Configuration
ROTOWIRE_URL = "https://www.rotowire.com/cbasketball/injury-report.php"
PROJECT_DIR = Path(
    r"C:\\Users\\omall\\Documents\\python_projects\\kenp0m-sp0rts-analyzer"
)
REPORTS_DIR = PROJECT_DIR / "reports" / "injury_reports"
LOG_DIR = PROJECT_DIR / "logs"

# Setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = (
    LOG_DIR / f"injury_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def verify_directory_structure():
    """
    Verify that the target directory exists, create if needed.

    Returns:
        Path: The verified target directory path
    """
    try:
        logger.info(f"Verifying directory structure...")

        # Check if project directory exists
        if not PROJECT_DIR.exists():
            logger.error(f"Project directory does not exist: {PROJECT_DIR}")
            return None

        # Create reports/injury_reports directory if it doesn't exist
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Target directory verified/created: {REPORTS_DIR}")

        return REPORTS_DIR

    except Exception as e:
        logger.error(f"Error verifying directory structure: {str(e)}")
        return None


def get_filename_with_date():
    """
    Generate filename with current date in YYYYMMDD format.

    Returns:
        str: Filename in format 'college-basketball-injury-report-YYYYMMDD.csv'
    """
    date_str = datetime.now().strftime("%Y%m%d")
    return f"college-basketball-injury-report-{date_str}.csv"


def download_injury_report_selenium():
    """
    Download the injury report using Selenium for JavaScript rendering.
    This approach handles the dynamic table content.

    Returns:
        list: List of dictionaries containing injury data
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options

        logger.info("Starting Selenium webdriver for RotoWire...")

        # Configure Chrome options for headless mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)

        try:
            driver.get(ROTOWIRE_URL)
            logger.info(f"Loaded page: {ROTOWIRE_URL}")

            # Wait for the table to load
            wait = WebDriverWait(driver, 10)
            table = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "tr.is-hidden"))
            )

            logger.info("Table loaded successfully")

            # Extract table data
            injury_data = []
            rows = driver.find_elements(By.CSS_SELECTOR, "tr[class*='tr']")

            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 5:
                        injury_record = {
                            "Player": cells[0].text.strip(),
                            "Team": cells[1].text.strip(),
                            "Position": cells[2].text.strip(),
                            "Injury": cells[3].text.strip(),
                            "Status": cells[4].text.strip(),
                        }
                        injury_data.append(injury_record)
                except Exception as e:
                    logger.debug(f"Error parsing row: {str(e)}")
                    continue

            logger.info(
                f"Successfully extracted {len(injury_data)} injury records"
            )
            return injury_data

        finally:
            driver.quit()

    except ImportError:
        logger.warning("Selenium not installed. Using fallback method.")
        return download_injury_report_requests()
    except Exception as e:
        logger.error(f"Error downloading report with Selenium: {str(e)}")
        return None


def download_injury_report_requests():
    """
    Fallback method to download injury report using requests library.
    Note: This may not capture all dynamic content.

    Returns:
        list: List of dictionaries containing injury data
    """
    try:
        import requests

        logger.info("Downloading injury report using requests...")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(ROTOWIRE_URL, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        logger.info("Page parsed successfully")

        injury_data = []

        # Find the table and extract rows
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            for row in rows[1:]:  # Skip header row
                cells = row.find_all("td")
                if len(cells) >= 5:
                    injury_record = {
                        "Player": cells[0].get_text(strip=True),
                        "Team": cells[1].get_text(strip=True),
                        "Position": cells[2].get_text(strip=True),
                        "Injury": cells[3].get_text(strip=True),
                        "Status": cells[4].get_text(strip=True),
                    }
                    injury_data.append(injury_record)

        logger.info(
            f"Successfully extracted {len(injury_data)} injury records"
        )
        return injury_data

    except ImportError:
        logger.error(
            "Required libraries not installed. Please install: requests, beautifulsoup4"
        )
        return None
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        return None


def save_injury_report_csv(injury_data, target_dir):
    """
    Save injury data to CSV file with automatic overwriting.

    Args:
        injury_data (list): List of injury records
        target_dir (Path): Target directory path

    Returns:
        Path: Path to saved file, or None if failed
    """
    try:
        if not injury_data:
            logger.error("No injury data to save")
            return None

        # Generate filename with current date
        filename = get_filename_with_date()
        filepath = target_dir / filename

        # Write to CSV (will overwrite if exists)
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = injury_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(injury_data)

        file_size = os.path.getsize(filepath)
        logger.info(f"Successfully saved injury report to: {filepath}")
        logger.info(f"File size: {file_size} bytes")
        logger.info(f"Total records: {len(injury_data)}")

        return filepath

    except Exception as e:
        logger.error(f"Error saving injury report: {str(e)}")
        return None


def save_execution_log(success, filepath=None, error_msg=None):
    """
    Save execution summary to JSON log for monitoring.

    Args:
        success (bool): Whether execution was successful
        filepath (Path): Path to saved CSV file
        error_msg (str): Error message if failed
    """
    try:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "filepath": str(filepath) if filepath else None,
            "error": error_msg,
            "target_directory": str(REPORTS_DIR),
        }

        summary_file = LOG_DIR / "last_execution.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Execution summary saved")

    except Exception as e:
        logger.error(f"Error saving execution log: {str(e)}")


def main():
    """
    Main execution function.
    """
    logger.info("=" * 60)
    logger.info("RotoWire Injury Report Downloader Started")
    logger.info("=" * 60)

    # Step 1: Verify directory structure
    target_dir = verify_directory_structure()
    if not target_dir:
        logger.error("Failed to verify directory structure. Exiting.")
        save_execution_log(False, error_msg="Directory verification failed")
        return False

    # Step 2: Download injury report
    logger.info("Downloading injury report from RotoWire...")
    injury_data = download_injury_report_selenium()

    if not injury_data:
        logger.error("Failed to download injury report. Exiting.")
        save_execution_log(False, error_msg="Download failed")
        return False

    # Step 3: Save to CSV with dynamic naming
    filepath = save_injury_report_csv(injury_data, target_dir)

    if not filepath:
        logger.error("Failed to save injury report. Exiting.")
        save_execution_log(False, error_msg="Save failed")
        return False

    # Step 4: Log successful execution
    save_execution_log(True, filepath=filepath)

    logger.info("=" * 60)
    logger.info("RotoWire Injury Report Downloader Completed Successfully")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        save_execution_log(False, error_msg=str(e))
        sys.exit(1)
