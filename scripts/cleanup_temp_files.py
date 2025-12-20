"""
Cleanup temporary files from testing and training
"""

import os
import shutil
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TemporaryFileManager:
    """Manage temporary files and cleanup"""

    def __init__(self, base_dir: Path = Path("data/temp")):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_old_files(self, days: int = 7, dry_run: bool = False):
        """
        Remove temporary files older than N days

        Args:
            days: Number of days to keep
            dry_run: If True, show what would be deleted without deleting
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0
        freed_space = 0

        for file_path in self.base_dir.rglob("*"):
            if file_path.is_file():
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)

                if mod_time < cutoff_time:
                    file_size = file_path.stat().st_size

                    if dry_run:
                        logger.info(
                            f"Would delete: {file_path} ({file_size} bytes)"
                        )
                    else:
                        try:
                            file_path.unlink()
                            logger.info(
                                f"Deleted: {file_path} ({file_size} bytes)"
                            )
                            deleted_count += 1
                            freed_space += file_size
                        except Exception as e:
                            logger.error(
                                f"Failed to delete {file_path}: {str(e)}"
                            )

        logger.info(
            f"Cleanup complete: {deleted_count} files deleted, {freed_space} bytes freed"
        )
        return {"deleted_count": deleted_count, "freed_space": freed_space}

    def cleanup_cache_directories(self, dry_run: bool = False):
        """Remove Python cache directories"""
        cache_patterns = [
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "*.pyc",
        ]

        deleted_count = 0

        for pattern in cache_patterns:
            if "*" in pattern:
                # Handle file patterns
                for file_path in Path(".").rglob(pattern):
                    if file_path.is_file():
                        if dry_run:
                            logger.info(f"Would delete: {file_path}")
                        else:
                            try:
                                file_path.unlink()
                                deleted_count += 1
                            except Exception as e:
                                logger.error(
                                    f"Failed to delete {file_path}: {str(e)}"
                                )
            else:
                # Handle directory patterns
                for dir_path in Path(".").rglob(pattern):
                    if dir_path.is_dir():
                        if dry_run:
                            logger.info(f"Would delete: {dir_path}")
                        else:
                            try:
                                shutil.rmtree(dir_path)
                                deleted_count += 1
                            except Exception as e:
                                logger.error(
                                    f"Failed to delete {dir_path}: {str(e)}"
                                )

        logger.info(f"Cache cleanup complete: {deleted_count} items deleted")
        return {"deleted_count": deleted_count}

    def cleanup_failed_logs(self, days: int = 7, dry_run: bool = False):
        """Remove old error logs"""
        logs_dir = Path("logs")
        error_log = logs_dir / "ncaab-xgboost_errors.log"

        if error_log.exists():
            mod_time = datetime.fromtimestamp(error_log.stat().st_mtime)
            cutoff_time = datetime.now() - timedelta(days=days)

            if mod_time < cutoff_time:
                if dry_run:
                    logger.info(f"Would delete error log: {error_log}")
                    return {"deleted_count": 0}
                else:
                    try:
                        error_log.unlink()
                        logger.info(f"Deleted error log: {error_log}")
                        return {"deleted_count": 1}
                    except Exception as e:
                        logger.error(f"Failed to delete error log: {str(e)}")
                        return {"deleted_count": 0}

        return {"deleted_count": 0}


def main():
    """Main cleanup routine"""
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    manager = TemporaryFileManager()

    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    days = 7

    # Parse days argument
    if "--days" in sys.argv:
        idx = sys.argv.index("--days")
        if idx + 1 < len(sys.argv):
            try:
                days = int(sys.argv[idx + 1])
            except ValueError:
                logger.warning(f"Invalid days value, using default: {days}")

    logger.info(f"Starting cleanup (dry_run={dry_run}, days={days})")

    # Run cleanup tasks
    logger.info("\\n=== Cleaning temporary files ===")
    manager.cleanup_old_files(days=days, dry_run=dry_run)

    logger.info("\\n=== Cleaning cache directories ===")
    manager.cleanup_cache_directories(dry_run=dry_run)

    logger.info("\\n=== Cleaning old error logs ===")
    manager.cleanup_failed_logs(days=days, dry_run=dry_run)

    logger.info("\\nCleanup process completed!")


if __name__ == "__main__":
    main()
