"""S3-compatible object storage with a local cache (write-through / read-through).

The pipeline's heavy lifting (rawpy, OpenCV) needs real files, so the local data
tree stays exactly as it is — this module just makes it durable:

  - STORAGE.put(path)   after producing a durable file  -> upload (write-through)
  - STORAGE.get(path)   before reading one that may be remote -> download if the
                        local cache is cold, return the path (or None if nowhere)
  - STORAGE.exists / mtime  answer from the local cache first, then a cached
                        remote listing (one LIST per TTL, not a HEAD per file)

Keys are the file's path relative to CONFIG.root (optionally under S3_PREFIX), so
`data/output/IMG_1.jpg` <-> `s3://bucket/[prefix/]output/IMG_1.jpg`. Works with any
S3 API: AWS, MinIO, Cloudflare R2, Backblaze B2, DigitalOcean Spaces…

Backend "local" (the default) turns every method into a no-op/local stat, so the
rest of the code calls STORAGE unconditionally and behaves exactly as before.

Config (env): PIPELINE_STORAGE=s3, S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
S3_BUCKET, S3_REGION, S3_PREFIX.
"""

import logging
import threading
import time
from pathlib import Path

from .config import CONFIG

log = logging.getLogger("storage")

_INDEX_TTL_S = 30.0          # how long the remote listing is trusted
_SCRATCH_DIRS = ("work",)    # never mirrored: intermediate TIFFs are rebuildable


class Storage:
    def __init__(self) -> None:
        self.backend = (CONFIG.storage_backend or "local").lower()
        self._client = None
        self._client_lock = threading.Lock()
        self._index: dict[str, float] | None = None   # key -> remote mtime (epoch)
        self._index_at = 0.0
        self._index_lock = threading.Lock()
        if self.backend == "s3" and not (CONFIG.s3_endpoint or CONFIG.s3_region):
            log.warning("PIPELINE_STORAGE=s3 but no S3_ENDPOINT_URL/S3_REGION configured")

    # ----------------------------------------------------------------- plumbing
    @property
    def enabled(self) -> bool:
        return self.backend == "s3"

    def client(self):
        """Lazy boto3 client so `local` deployments don't need boto3 at all."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import boto3
                    from botocore.config import Config as BotoConfig
                    self._client = boto3.client(
                        "s3",
                        endpoint_url=CONFIG.s3_endpoint or None,
                        aws_access_key_id=CONFIG.s3_access_key or None,
                        aws_secret_access_key=CONFIG.s3_secret_key or None,
                        region_name=CONFIG.s3_region or None,
                        config=BotoConfig(s3={"addressing_style": "path"},
                                          retries={"max_attempts": 3}),
                    )
        return self._client

    def key_for(self, path: Path) -> str | None:
        """Object key for a file under CONFIG.root; None if outside or scratch."""
        try:
            rel = Path(path).resolve().relative_to(CONFIG.root.resolve())
        except ValueError:
            return None
        if rel.parts and rel.parts[0] in _SCRATCH_DIRS:
            return None
        key = rel.as_posix()
        return f"{CONFIG.s3_prefix}/{key}" if CONFIG.s3_prefix else key

    def _remote_index(self, refresh: bool = False) -> dict[str, float]:
        """key -> mtime for every object under the prefix, cached for _INDEX_TTL_S."""
        with self._index_lock:
            if (not refresh and self._index is not None
                    and time.time() - self._index_at < _INDEX_TTL_S):
                return self._index
            index: dict[str, float] = {}
            paginator = self.client().get_paginator("list_objects_v2")
            kwargs = {"Bucket": CONFIG.s3_bucket}
            if CONFIG.s3_prefix:
                kwargs["Prefix"] = CONFIG.s3_prefix + "/"
            for page in paginator.paginate(**kwargs):
                for obj in page.get("Contents", []):
                    index[obj["Key"]] = obj["LastModified"].timestamp()
            self._index, self._index_at = index, time.time()
            return index

    def _note_uploaded(self, key: str, mtime: float) -> None:
        with self._index_lock:
            if self._index is not None:
                self._index[key] = mtime

    # ----------------------------------------------------------------- operations
    def put(self, path: Path) -> None:
        """Upload a freshly produced file (keeps the local copy as cache)."""
        if not self.enabled:
            return
        key = self.key_for(path)
        if key is None or not Path(path).exists():
            return
        try:
            self.client().upload_file(str(path), CONFIG.s3_bucket, key)
            self._note_uploaded(key, time.time())
            log.debug("Uploaded %s", key)
        except Exception as exc:      # storage must never break a render
            log.error("Upload of %s failed: %s", key, exc)

    def get(self, path: Path) -> Path | None:
        """Return a local path for the file, downloading it if the cache is cold.
        None when the file exists neither locally nor remotely."""
        path = Path(path)
        if path.exists():
            return path
        if not self.enabled:
            return None
        key = self.key_for(path)
        if key is None:
            return None
        index = self._remote_index()
        if key not in index:
            index = self._remote_index(refresh=True)   # might be brand new
            if key not in index:
                return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".s3tmp")
            self.client().download_file(CONFIG.s3_bucket, key, str(tmp))
            tmp.replace(path)
            log.info("Restored %s from object storage", key)
            return path
        except Exception as exc:
            log.error("Download of %s failed: %s", key, exc)
            return None

    def exists(self, path: Path) -> bool:
        if Path(path).exists():
            return True
        if not self.enabled:
            return False
        key = self.key_for(path)
        return key is not None and key in self._remote_index()

    def mtime(self, path: Path) -> float:
        """Modification time from the cache, else from the remote listing, else 0."""
        p = Path(path)
        if p.exists():
            return p.stat().st_mtime
        if self.enabled:
            key = self.key_for(p)
            if key is not None:
                return self._remote_index().get(key, 0.0)
        return 0.0

    def delete(self, path: Path) -> None:
        """Remove a file locally and remotely (used for cleared erase masks)."""
        Path(path).unlink(missing_ok=True)
        if not self.enabled:
            return
        key = self.key_for(path)
        if key is None:
            return
        try:
            self.client().delete_object(Bucket=CONFIG.s3_bucket, Key=key)
            with self._index_lock:
                if self._index is not None:
                    self._index.pop(key, None)
        except Exception as exc:
            log.error("Delete of %s failed: %s", key, exc)


STORAGE = Storage()
