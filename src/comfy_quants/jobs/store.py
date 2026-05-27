"""File-based job store."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comfy_quants.core.checkpoint import CheckpointRecord
from comfy_quants.utils.jsonio import read_json, write_json


@dataclass
class JobRecord:
    """Persistent top-level job record."""

    job_id: str
    status: str
    work_dir: str
    config_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    current_step: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    """Simple directory-backed job store."""

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def job_path(self) -> Path:
        return self.work_dir / "job.json"

    @property
    def checkpoints_dir(self) -> Path:
        path = self.work_dir / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create(self, job_id: str, status: str = "created", config_path: str | None = None, **metadata: Any) -> JobRecord:
        record = JobRecord(job_id=job_id, status=status, work_dir=str(self.work_dir), config_path=config_path, metadata=metadata)
        self.save(record)
        return record

    def load(self) -> JobRecord:
        data = read_json(self.job_path)
        return JobRecord(**data)

    def save(self, record: JobRecord) -> None:
        record.updated_at = datetime.now(timezone.utc).isoformat()
        write_json(self.job_path, record.to_dict())

    def set_status(self, status: str, message: str = "", current_step: str | None = None) -> JobRecord:
        record = self.load()
        record.status = status
        record.message = message
        record.current_step = current_step
        self.save(record)
        return record

    def write_checkpoint(self, checkpoint: CheckpointRecord) -> Path:
        path = self.checkpoints_dir / f"{checkpoint.step_id}_{checkpoint.status}.json"
        write_json(path, checkpoint.to_dict())
        return path


def list_jobs(root: str | Path) -> list[dict[str, Any]]:
    """Find job.json files under a root directory."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    jobs: list[dict[str, Any]] = []
    for path in root_path.rglob("job.json"):
        try:
            jobs.append(read_json(path))
        except Exception:
            continue
    return jobs
