"""Atomic local write-ahead evidence state alongside GEPA's native checkpoint."""

from __future__ import annotations

import errno
import json
import os
import tempfile
from typing import Any

from gepa.evidence.config import canonical_json, digest


class EvidenceStore:
    SCHEMA_VERSION = 1

    def __init__(self, run_dir: str | None, context: str) -> None:
        self.run_dir = run_dir
        self.context = context

    @property
    def path(self) -> str | None:
        return os.path.join(self.run_dir, "sequential_evidence.json") if self.run_dir is not None else None

    def load(self) -> dict[str, Any] | None:
        path = self.path
        if path is None or not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as stream:
                envelope = json.load(stream)
            if envelope["schema_version"] != self.SCHEMA_VERSION:
                raise ValueError("unsupported sequential evidence schema version")
            if envelope["context"] != self.context:
                raise ValueError("sequential evidence context mismatch; use a different run_dir")
            payload = envelope["payload"]
            if digest(payload) != envelope["checksum"]:
                raise ValueError("sequential evidence checksum mismatch")
            if not isinstance(payload, dict):
                raise ValueError("invalid sequential evidence payload")
            return payload
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("corrupt sequential evidence artifact") from error

    def save(self, payload: dict[str, Any]) -> None:
        path = self.path
        if path is None:
            return
        assert self.run_dir is not None
        os.makedirs(self.run_dir, exist_ok=True)
        envelope = {
            "schema_version": self.SCHEMA_VERSION,
            "context": self.context,
            "payload": payload,
            "checksum": digest(payload),
        }
        serialized = canonical_json(envelope)
        descriptor, temporary = tempfile.mkstemp(prefix=".sequential-evidence-", dir=self.run_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            # Persist the rename where directory fsync is supported.
            if os.name != "nt":
                directory = os.open(self.run_dir, os.O_RDONLY)
                try:
                    try:
                        os.fsync(directory)
                    except OSError as error:
                        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
                            raise
                finally:
                    os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
