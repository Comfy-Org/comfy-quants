"""jobs and resume commands."""

from __future__ import annotations

from pathlib import Path

from comfy_quants.cli.common import print_json
from comfy_quants.jobs.store import JobStore, list_jobs


def register_jobs(subparsers):
    parser = subparsers.add_parser("jobs", help="Inspect Comfy Quants jobs")
    job_sub = parser.add_subparsers(dest="jobs_command", required=True)
    list_parser = job_sub.add_parser("list", help="List jobs under a root directory")
    list_parser.add_argument("--work-dir", default="runs")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=run_list)
    status_parser = job_sub.add_parser("status", help="Show a job.json status")
    status_parser.add_argument("job", help="Path to job.json or job work directory")
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=run_status)


def register_resume(subparsers):
    parser = subparsers.add_parser("resume", help="Resume a quantization job")
    parser.add_argument("job", help="Path to job.json or job work directory")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=run_resume)


def _store_from_arg(job_arg: str) -> JobStore:
    p = Path(job_arg)
    return JobStore(p.parent if p.name == "job.json" else p)


def run_list(args) -> int:
    jobs = list_jobs(args.work_dir)
    if args.json:
        print_json({"jobs": jobs})
    else:
        for job in jobs:
            print(f"{job.get('job_id')}\t{job.get('status')}\t{job.get('work_dir')}")
    return 0


def run_status(args) -> int:
    record = _store_from_arg(args.job).load().to_dict()
    if args.json:
        print_json(record)
    else:
        print(f"job_id={record['job_id']} status={record['status']} work_dir={record['work_dir']} message={record.get('message','')}")
    return 0


def run_resume(args) -> int:
    store = _store_from_arg(args.job)
    record = store.set_status("resume_requested", "resume request recorded")
    if args.json:
        print_json(record.to_dict())
    else:
        print(f"resume requested for {record.job_id}")
    return 0
