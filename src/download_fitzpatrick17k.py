"""Download Fitzpatrick17k images from the source URLs in the metadata CSV.

Each image is saved as <md5hash>.jpg, the naming the dataset's own train.py uses.
Files that already exist are skipped, so the script can be re-run to resume.
Every row that could not be downloaded is written to a failures CSV, which is
rewritten on each run so it only lists what is still missing.

Usage:
    python src/download_fitzpatrick17k.py
    python src/download_fitzpatrick17k.py --delay 1.0 --retries 3
"""

import argparse
import csv
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "fitzpatrick"
DEFAULT_CSV = DATA_DIR / "fitzpatrick17k.csv"
DEFAULT_OUT = DATA_DIR / "images"
DEFAULT_FAILURES = DATA_DIR / "download_failures.csv"
USER_AGENT = "Mozilla/5.0 (research dataset download)"
TIMEOUT_SECONDS = 30
# Statuses worth retrying; a 404 or 403 will not change on a second try.
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
# python.org builds on macOS ship without CA certificates, so fall back to the
# system bundle. Certificate verification stays on.
SYSTEM_CA_BUNDLE = Path("/etc/ssl/cert.pem")
SSL_CONTEXT = ssl.create_default_context(
    cafile=str(SYSTEM_CA_BUNDLE) if SYSTEM_CA_BUNDLE.exists() else None
)


def fetch(url: str) -> bytes:
    """Return the image bytes, or raise ValueError/HTTPError/URLError."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS, context=SSL_CONTEXT) as response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read()
    if not content_type.startswith("image/"):
        raise ValueError(f"not an image (Content-Type: {content_type or 'missing'})")
    if not body:
        raise ValueError("empty response")
    return body


def download(url: str, retries: int, delay: float) -> tuple[bytes | None, str, str]:
    """Try to download url. Returns (bytes or None, status, error message)."""
    status, error = "", ""
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(delay * 2**attempt)  # back off before each retry
        try:
            return fetch(url), "200", ""
        except urllib.error.HTTPError as exc:
            status, error = str(exc.code), exc.reason or "HTTP error"
            if exc.code not in RETRYABLE_STATUSES:
                break
        except ValueError as exc:
            status, error = "bad_content", str(exc)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            status, error = "network_error", str(getattr(exc, "reason", exc))
    return None, status, error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="seconds to wait between requests (default 0.5)")
    parser.add_argument("--retries", type=int, default=2,
                        help="extra attempts after a timeout or server error (default 2)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures = []
    for i, row in enumerate(rows, start=1):
        md5, url = row["md5hash"], row["url"].strip()
        target = args.out_dir / f"{md5}.jpg"
        if target.exists():
            counts["skipped"] += 1
            continue

        if not url:
            data, status, error = None, "no_url", "url column is empty"
        else:
            data, status, error = download(url, args.retries, args.delay)
            time.sleep(args.delay)

        if data is None:
            counts["failed"] += 1
            failures.append({"md5hash": md5, "label": row["label"], "url": url,
                             "status": status, "error": error})
        else:
            partial = target.with_suffix(".part")  # avoid half-written .jpg files
            partial.write_bytes(data)
            partial.replace(target)
            counts["downloaded"] += 1

        if i % 250 == 0:
            print(f"[{i}/{len(rows)}] {counts}", flush=True)

    with args.failures.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["md5hash", "label", "url", "status", "error"])
        writer.writeheader()
        writer.writerows(failures)

    print(f"Done. {counts}. Images in {args.out_dir}. Failures logged to {args.failures}.")


if __name__ == "__main__":
    main()
