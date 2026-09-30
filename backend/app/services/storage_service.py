from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile
import urllib.error
import urllib.request
from typing import Any

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class StorageCleanupResult(str, Enum):
    """Classification of an object storage cleanup attempt."""
    DELETED = "deleted"                  # 200 / 204: Object successfully deleted
    ALREADY_ABSENT = "already_absent"    # 404, or 400 with 'not found': Object already gone
    INVALID_PATH = "invalid_path"        # Malformed path, traversal attempt, or non-model path
    PERMISSION_DENIED = "permission_denied"  # 401 / 403: Authorization or RLS failure
    TRANSIENT_ERROR = "transient_error"  # 5xx, timeouts, connection dropouts


@dataclass(frozen=True)
class DeleteResult:
    """Result metadata for an artifact deletion operation."""
    path: str
    result: StorageCleanupResult
    status_code: int | None = None
    message: str | None = None

    @property
    def is_success_or_benign(self) -> bool:
        return self.result in (StorageCleanupResult.DELETED, StorageCleanupResult.ALREADY_ABSENT)


def validate_artifact_path(path: str) -> bool:
    """
    Validate that an artifact path adheres strictly to internal model storage conventions
    and contains no path traversal ('..') or dangerous characters.

    Expected format: models/{user_id}/{dataset_id}/v{version}/{horizon_or_type}/model.joblib
    """
    if not path or not isinstance(path, str):
        return False
    clean = path.strip().lstrip("/\\")
    if not clean:
        return False
    parts = clean.replace("\\", "/").split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    if parts[0] != "models":
        return False
    if len(parts) < 5 or len(parts) > 7:
        return False
    filename = parts[-1]
    if not (filename.endswith(".joblib") or filename.endswith(".json")):
        return False
    return True


def build_model_artifact_path(
    user_id: int | str,
    dataset_id: str,
    model_version: int | str,
    horizon_or_type: int | str,
) -> str:
    """
    Build canonical storage path for a model artifact.
    Example: models/1/ds-xyz/v1/7/model.joblib or models/1/ds-xyz/v1/elasticity/model.joblib
    """
    v_clean = str(model_version).lstrip("v")
    h_clean = str(horizon_or_type)
    return f"models/{user_id}/{dataset_id}/v{v_clean}/{h_clean}/model.joblib"


def classify_supabase_storage_error(status_code: int, err_body: str = "") -> StorageCleanupResult:
    """
    Classify Supabase Storage HTTP error status code and response body
    into a structured domain cleanup result.
    """
    body_lower = (err_body or "").lower()

    # 1. Check for absent object indicators
    # Supabase Storage returns 404, or 400 with 'not found' / 'resource was not found' when an object is absent
    absent_indicators = [
        "not found",
        "resource was not found",
        "resource not found",
        "object not found",
        "not_found",
        "nosuchkey",
        "does not exist",
        "no such file",
    ]
    if status_code == 404:
        return StorageCleanupResult.ALREADY_ABSENT

    if status_code == 400:
        if any(ind in body_lower for ind in absent_indicators):
            return StorageCleanupResult.ALREADY_ABSENT
        # Any other 400 is NOT classified as absent; it remains an actionable failure.
        return StorageCleanupResult.INVALID_PATH

    if status_code in (401, 403) or "unauthorized" in body_lower or "forbidden" in body_lower or "jwt" in body_lower:
        return StorageCleanupResult.PERMISSION_DENIED

    if status_code >= 500:
        return StorageCleanupResult.TRANSIENT_ERROR

    return StorageCleanupResult.TRANSIENT_ERROR


class StorageError(Exception):
    """Base exception for storage backend operations."""
    pass


class StorageFileNotFoundError(StorageError):
    """Raised when an artifact cannot be located in storage."""
    pass


class StorageConfigurationError(StorageError):
    """Raised when required storage credentials or configurations are missing."""
    pass


class StorageBackend(ABC):
    """Abstract interface for artifact object storage."""

    @abstractmethod
    def upload(self, remote_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload raw binary data to remote path and return the URI or path."""
        pass

    @abstractmethod
    def download(self, remote_path: str) -> bytes:
        """Download and return raw binary data from remote path."""
        pass

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if remote path exists in storage."""
        pass

    @abstractmethod
    def delete(self, remote_path: str) -> DeleteResult:
        """Delete object at remote path if present and return DeleteResult."""
        pass


class LocalStorageBackend(StorageBackend):
    """
    Filesystem-backed storage for development, testing, and offline environments.
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        if base_dir:
            self.base_dir = Path(base_dir)
        elif settings.model_storage_local_dir:
            self.base_dir = Path(settings.model_storage_local_dir)
        else:
            self.base_dir = PROJECT_ROOT / "storage" / "models"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, remote_path: str) -> Path:
        clean = remote_path.lstrip("/\\")
        return self.base_dir / clean

    def upload(self, remote_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        target = self._resolve(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via temp file in same directory
        temp_target = target.with_suffix(".tmp")
        with open(temp_target, "wb") as f:
            f.write(data)
        os.replace(temp_target, target)
        return str(remote_path)

    def download(self, remote_path: str) -> bytes:
        target = self._resolve(remote_path)
        if not target.exists():
            raise StorageFileNotFoundError(f"Local artifact not found: {target}")
        with open(target, "rb") as f:
            return f.read()

    def exists(self, remote_path: str) -> bool:
        return self._resolve(remote_path).exists()

    def delete(self, remote_path: str) -> DeleteResult:
        clean_path = remote_path.strip().lstrip("/\\") if remote_path else ""
        if not validate_artifact_path(clean_path):
            logger.warning("local_storage_cleanup_invalid_path: path=%s", clean_path)
            return DeleteResult(
                path=clean_path,
                result=StorageCleanupResult.INVALID_PATH,
                message="Path failed artifact validation constraints",
            )
        target = self._resolve(clean_path)
        if not target.exists():
            logger.info("local_storage_cleanup_already_absent: path=%s", clean_path)
            return DeleteResult(
                path=clean_path,
                result=StorageCleanupResult.ALREADY_ABSENT,
            )
        try:
            target.unlink()
            logger.info("local_storage_cleanup_success: path=%s", clean_path)
            return DeleteResult(
                path=clean_path,
                result=StorageCleanupResult.DELETED,
                status_code=200,
            )
        except OSError as exc:
            logger.warning("local_storage_cleanup_failed: path=%s, error=%s", clean_path, exc)
            return DeleteResult(
                path=clean_path,
                result=StorageCleanupResult.TRANSIENT_ERROR,
                message=str(exc),
            )


class SupabaseStorageBackend(StorageBackend):
    """
    Object storage backend integrating with Supabase Storage REST API.
    Uses standard library urllib for zero external dependencies.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        bucket_name: str = "company-models",
    ) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.bucket = bucket_name

    def _get_headers(self, content_type: str = "application/octet-stream", upsert: bool = False) -> dict[str, str]:
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": content_type,
        }
        if upsert:
            headers["x-upsert"] = "true"
        return headers

    def _ensure_bucket(self) -> None:
        """Attempt to auto-create the private Supabase storage bucket if it does not already exist."""
        import json
        url = f"{self.supabase_url}/storage/v1/bucket"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "id": self.bucket,
            "name": self.bucket,
            "public": False,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status in (200, 201):
                    logger.info("Successfully created Supabase storage bucket '%s'", self.bucket)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.debug("Supabase bucket creation response (%s): %s", exc.code, body)
        except Exception as exc:
            logger.warning("Could not auto-create Supabase storage bucket '%s': %s", self.bucket, exc)

    def upload(self, remote_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        clean_path = remote_path.lstrip("/\\")
        url = f"{self.supabase_url}/storage/v1/object/{self.bucket}/{clean_path}"
        headers = self._get_headers(content_type=content_type, upsert=True)

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status not in (200, 201):
                    raise StorageError(f"Supabase upload returned HTTP {resp.status}")
                return clean_path
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            # If the bucket does not exist, attempt auto-creation using service credentials and retry upload once
            if exc.code in (400, 404) and ("NoSuchBucket" in err_body or "Bucket not found" in err_body):
                logger.info("Supabase bucket '%s' not found. Attempting auto-creation...", self.bucket)
                self._ensure_bucket()
                try:
                    retry_req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                    with urllib.request.urlopen(retry_req, timeout=30) as retry_resp:
                        if retry_resp.status in (200, 201):
                            return clean_path
                except Exception as retry_exc:
                    logger.warning("Retry upload after bucket auto-creation failed: %s", retry_exc)
            raise StorageError(f"Supabase upload failed ({exc.code}): {err_body}") from exc
        except Exception as exc:
            raise StorageError(f"Network error during Supabase upload: {exc}") from exc

    def download(self, remote_path: str) -> bytes:
        clean_path = remote_path.lstrip("/\\")
        url = f"{self.supabase_url}/storage/v1/object/authenticated/{self.bucket}/{clean_path}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
        }

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    raise StorageError(f"Supabase download returned HTTP {resp.status}")
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise StorageFileNotFoundError(f"Artifact not found on Supabase: {clean_path}") from exc
            err_body = exc.read().decode("utf-8", errors="replace")
            raise StorageError(f"Supabase download failed ({exc.code}): {err_body}") from exc
        except Exception as exc:
            raise StorageError(f"Network error during Supabase download: {exc}") from exc

    def exists(self, remote_path: str) -> bool:
        try:
            self.download(remote_path)
            return True
        except StorageFileNotFoundError:
            return False
        except Exception:
            return False

    def delete(self, remote_path: str) -> DeleteResult:
        clean_path = remote_path.strip().lstrip("/\\") if remote_path else ""
        if not validate_artifact_path(clean_path):
            logger.warning(
                "storage_cleanup_invalid_path: bucket=%s, path=%s",
                self.bucket,
                clean_path,
            )
            return DeleteResult(
                path=clean_path,
                result=StorageCleanupResult.INVALID_PATH,
                message="Path failed artifact validation constraints",
            )

        url = f"{self.supabase_url}/storage/v1/object/{self.bucket}/{clean_path}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
        }
        req = urllib.request.Request(url, headers=headers, method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = getattr(resp, "status", 200)
                logger.info(
                    "storage_cleanup_success: bucket=%s, path=%s, status=%s",
                    self.bucket,
                    clean_path,
                    status,
                )
                return DeleteResult(
                    path=clean_path,
                    result=StorageCleanupResult.DELETED,
                    status_code=status,
                )
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            classified = classify_supabase_storage_error(exc.code, err_body)

            if classified == StorageCleanupResult.ALREADY_ABSENT:
                logger.info(
                    "storage_cleanup_already_absent: bucket=%s, path=%s, status=%s",
                    self.bucket,
                    clean_path,
                    exc.code,
                )
            elif classified == StorageCleanupResult.PERMISSION_DENIED:
                logger.error(
                    "storage_cleanup_permission_denied: bucket=%s, path=%s, status=%s",
                    self.bucket,
                    clean_path,
                    exc.code,
                )
            elif classified == StorageCleanupResult.INVALID_PATH:
                logger.warning(
                    "storage_cleanup_rejected_by_storage: bucket=%s, path=%s, status=%s",
                    self.bucket,
                    clean_path,
                    exc.code,
                )
            else:
                logger.warning(
                    "storage_cleanup_transient_http_error: bucket=%s, path=%s, status=%s",
                    self.bucket,
                    clean_path,
                    exc.code,
                )

            return DeleteResult(
                path=clean_path,
                result=classified,
                status_code=exc.code,
                message=err_body[:200],
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "storage_cleanup_network_error: bucket=%s, path=%s, error=%s",
                self.bucket,
                clean_path,
                type(exc).__name__,
            )
            return DeleteResult(
                path=clean_path,
                result=StorageCleanupResult.TRANSIENT_ERROR,
                message=f"{type(exc).__name__}: {str(exc)}",
            )
        except Exception as exc:
            logger.warning(
                "storage_cleanup_unexpected_error: bucket=%s, path=%s, error=%s",
                self.bucket,
                clean_path,
                type(exc).__name__,
            )
            return DeleteResult(
                path=clean_path,
                result=StorageCleanupResult.TRANSIENT_ERROR,
                message=str(exc),
            )


class ModelStorageService:
    """
    High-level model storage manager.
    Handles pluggable backends and local container disk caching to ensure fast inference
    and survival across Render ephemeral container restarts.
    """

    def __init__(self, backend: StorageBackend | None = None) -> None:
        if backend is not None:
            self.backend = backend
        else:
            self.backend = self._init_default_backend()

        self.cache_dir = Path(tempfile.gettempdir()) / "sales_models_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_default_backend(self) -> StorageBackend:
        sb_url = settings.supabase_url
        # Privileged service role key is mandatory in production
        sb_key = settings.supabase_service_role_key
        if not sb_key and not settings.is_production:
            # Fall back to supabase_key only in development/testing
            sb_key = settings.supabase_key

        if sb_url and sb_key:
            logger.info("Initializing SupabaseStorageBackend for model artifacts (bucket: %s)", settings.supabase_storage_bucket)
            return SupabaseStorageBackend(
                supabase_url=sb_url,
                supabase_key=sb_key,
                bucket_name=settings.supabase_storage_bucket,
            )
        if settings.is_production:
            raise StorageConfigurationError(
                "Production environment requires privileged Supabase Object Storage credentials. "
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured. "
                "Public/anon keys are not permitted for private model artifacts in production."
            )
        logger.info("Supabase storage credentials not configured; defaulting to LocalStorageBackend.")
        return LocalStorageBackend()

    def get_local_cached_path(self, remote_path: str) -> Path:
        """Resolve expected path in the container disk cache."""
        clean = remote_path.lstrip("/\\")
        return self.cache_dir / clean

    def save_artifact(self, remote_path: str, data: bytes) -> str:
        """
        Write artifact to local container cache atomically and upload to persistent storage.
        """
        # 1. Write to container disk cache
        local_path = self.get_local_cached_path(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_local = local_path.with_suffix(".tmp")
        with open(tmp_local, "wb") as f:
            f.write(data)
        os.replace(tmp_local, local_path)

        # 2. Upload to persistent storage backend
        self.backend.upload(remote_path, data)
        return remote_path

    def load_artifact_file(self, remote_path: str) -> Path:
        """
        Return the local Path to the artifact file.
        Checks container disk cache first; downloads from persistent storage on cache miss.
        """
        local_path = self.get_local_cached_path(remote_path)
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path

        # Cache miss: download from storage backend
        logger.info("Artifact container cache miss for %s. Fetching from storage backend...", remote_path)
        data = self.backend.download(remote_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_local = local_path.with_suffix(".tmp")
        with open(tmp_local, "wb") as f:
            f.write(data)
        os.replace(tmp_local, local_path)

        return local_path

    def delete_artifact(self, remote_path: str) -> DeleteResult:
        """Purge from container cache and delete from remote storage."""
        clean = remote_path.strip().lstrip("/\\") if remote_path else ""
        if clean:
            local_path = self.get_local_cached_path(clean)
            if local_path.exists():
                try:
                    local_path.unlink()
                except OSError:
                    pass
        return self.backend.delete(remote_path)


storage_service = ModelStorageService()
