"""Fetch the ambiguity-aware (cIMLE) prior checkpoint.

The deterministic prior needs no weights of its own, but the ambiguity-aware
one - the prior that produces several albedo samples per view, which the space
carving loss consumes - is a checkpoint we publish as a release asset.

    python tools/download_prior_checkpoint.py
    python tools/download_prior_checkpoint.py --out ./checkpoints/cimle_yagiz_v1.pt

Then pass the file to the extraction tool:

    --prior_model cIMLE_yagiz_v1 --model_checkpoint ./checkpoints/cimle_yagiz_v1.pt
"""

import argparse
import hashlib
import os
import urllib.request

URL = (
    "https://github.com/amoazeni75/Intrinsic-PAPR/releases/download/"
    "v1.0/cimle_yagiz_v1.pt"
)
SHA256 = "9e82a31d16de1c72e79bb4cc"  # first 24 hex chars, checked as a prefix
DEFAULT_OUT = os.path.join("checkpoints", "cimle_yagiz_v1.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the file is there"
    )
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        print("Already present: {}".format(args.out))
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print("Downloading {}\n        -> {}".format(URL, args.out))
    urllib.request.urlretrieve(URL, args.out)

    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    if not digest.startswith(SHA256):
        raise SystemExit(
            "Checksum mismatch: expected {}..., got {}...".format(SHA256, digest[:24])
        )
    print("Done ({:.0f} MB), checksum ok.".format(os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
