from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
import urllib.error
import urllib.request
from typing import Any

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    def delete(self, remote_path: str) -> None:
        """Delete object at remote path if present."""
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

    def delete(self, remote_path: str) -> None:
        target = self._resolve(remote_path)
        if target.exists():
            try:
                target.unlink()
            except OSError as exc:
                logger.warning("Failed to delete local artifact %s: %s", target, exc)


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

    def delete(self, remote_path: str) -> None:
        clean_path = remote_path.lstrip("/\\")
        url = f"{self.supabase_url}/storage/v1/object/{self.bucket}/{clean_path}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
        }
        req = urllib.request.Request(url, headers=headers, method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                pass
        except Exception as exc:
            logger.warning("Failed to delete remote Supabase object %s: %s", clean_path, exc)


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

    def delete_artifact(self, remote_path: str) -> None:
        """Purge from container cache and delete from remote storage."""
        local_path = self.get_local_cached_path(remote_path)
        if local_path.exists():
            try:
                local_path.unlink()
            except OSError:
                pass
        self.backend.delete(remote_path)


storage_service = ModelStorageService()
