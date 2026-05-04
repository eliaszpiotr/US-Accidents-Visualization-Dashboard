from pathlib import Path
import sys

import gdown


FILE_ID = "1EUbQjP8R8P1QbDuNFtfP8-TAjfqqJSeA"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TARGET_PATH = DATA_DIR / "filtereddata.parquet"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_PATH.exists():
        print(f"{TARGET_PATH.relative_to(PROJECT_ROOT)} already exists. Skipping download.")
        return 0

    print("Downloading filtereddata.parquet from Google Drive...")
    try:
        downloaded_path = gdown.download(
            id=FILE_ID,
            output=str(TARGET_PATH),
            quiet=False,
            fuzzy=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to download dataset from Google Drive. "
            "Check your internet connection and file permissions."
        ) from exc

    if not downloaded_path or not TARGET_PATH.exists() or TARGET_PATH.stat().st_size == 0:
        raise RuntimeError(
            "Download did not produce data/filtereddata.parquet. "
            "Try running the script again."
        )

    print(f"Saved to {TARGET_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
