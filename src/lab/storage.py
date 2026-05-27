"""Durable artifact storage on Cloudflare R2 (S3-compatible) — the canonical store so artifacts
survive instance teardown (research/15, FR-E).

Optional: enabled only when ``LAB_R2_ENDPOINT`` is set and credentials are available (the
``~/.cloudflare/r2.credentials`` AWS-format file, or AWS_* env vars). When disabled, the lab falls
back to local-only artifacts. Config (env):

    LAB_R2_ENDPOINT   https://<account>.r2.cloudflarestorage.com   (required to enable)
    LAB_R2_BUCKET     bucket name (default "lab-artifacts")
"""

from __future__ import annotations

import os
from pathlib import Path

R2_CREDENTIALS_FILE = Path.home() / ".cloudflare" / "r2.credentials"
DEFAULT_BUCKET = "lab-artifacts"


def r2_enabled() -> bool:
    """True if R2 is configured (endpoint + some credential source)."""
    if not os.environ.get("LAB_R2_ENDPOINT"):
        return False
    return bool(os.environ.get("AWS_ACCESS_KEY_ID")) or R2_CREDENTIALS_FILE.exists()


class R2Store:
    def __init__(self, endpoint: str, bucket: str = DEFAULT_BUCKET) -> None:
        import boto3

        # Let boto3 read the R2 creds file unless AWS creds are already in the env.
        if not os.environ.get("AWS_ACCESS_KEY_ID") and R2_CREDENTIALS_FILE.exists():
            os.environ.setdefault("AWS_SHARED_CREDENTIALS_FILE", str(R2_CREDENTIALS_FILE))
        self.bucket = bucket
        self._s3 = boto3.client("s3", endpoint_url=endpoint, region_name="auto")

    @classmethod
    def from_env(cls) -> R2Store | None:
        endpoint = os.environ.get("LAB_R2_ENDPOINT")
        if not endpoint:
            return None
        return cls(endpoint, os.environ.get("LAB_R2_BUCKET", DEFAULT_BUCKET))

    def uri(self, prefix: str) -> str:
        return f"r2://{self.bucket}/{prefix}"

    def upload_dir(self, local_dir: Path, prefix: str) -> int:
        """Upload every file under ``local_dir`` to ``<bucket>/<prefix>/...``; returns the count."""
        local_dir = Path(local_dir)
        count = 0
        for f in sorted(local_dir.rglob("*")):
            if f.is_file():
                key = f"{prefix}/{f.relative_to(local_dir).as_posix()}"
                self._s3.upload_file(str(f), self.bucket, key)
                count += 1
        return count

    def download_dir(self, prefix: str, local_dir: Path) -> int:
        """Download ``<bucket>/<prefix>/...`` into ``local_dir``; returns the count."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        token: str | None = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": f"{prefix}/"}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                rel = obj["Key"][len(prefix) + 1 :]
                if not rel:
                    continue
                dest = local_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(self.bucket, obj["Key"], str(dest))
                count += 1
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                return count
