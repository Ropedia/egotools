"""Download configured EgoTools resources from Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Resource:
    repo_id: str
    repo_type: str
    subdir: str | None
    status: str
    ignore_patterns: tuple[str, ...] = ()
    include_patterns: tuple[str, ...] = ()


def load_resources(config: Path | None = None) -> dict[str, Resource]:
    """Use the checked-in mapping, or its copy included in an installed wheel."""
    source_config = Path(__file__).resolve().parents[2] / "configs" / "resources.yaml"
    if config is not None or source_config.is_file():
        text = (config or source_config).read_text(encoding="utf-8")
    else:
        text = files("egotools").joinpath("resources.yaml").read_text(encoding="utf-8")
    return {
        name.replace("_", "-"): Resource(
            repo_id=value["repo_id"], repo_type=value["repo_type"],
            subdir=value.get("subdir"), status=value.get("status", "custom"),
            ignore_patterns=tuple(value.get("ignore_patterns", [])),
            include_patterns=tuple(value.get("include_patterns", [])),
        )
        for name, value in yaml.safe_load(text).items()
    }


RESOURCES = load_resources()


def resolve_resource(
    name: str,
    *,
    repo_id: str | None = None,
    repo_type: str | None = None,
    subdir: str | None = None,
    config: Path | None = None,
) -> Resource:
    resource = load_resources(config)[name]
    if repo_id is not None:
        resource = replace(resource, repo_id=repo_id, status="custom", ignore_patterns=(), include_patterns=())
    if repo_type is not None:
        resource = replace(resource, repo_type=repo_type)
    if subdir is not None:
        resource = replace(resource, subdir=subdir or None, include_patterns=())
    return resource


def download_resource(
    resource: Resource,
    *,
    output_dir: Path,
    revision: str | None = None,
    metadata_only: bool = False,
) -> Path:
    """Download a resource and return its concrete local content path."""

    if not resource.repo_id:
        raise ValueError(
            "The final resource repository is not configured yet. "
            "Pass --repo-id with a confirmed repository ID, or explicitly select "
            "--config configs/resources.preview.yaml for historical resources."
        )

    from huggingface_hub import snapshot_download

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [f"{resource.subdir}/**"] if resource.subdir else None
    if resource.include_patterns:
        allow_patterns = list(resource.include_patterns)
    if metadata_only:
        prefix = f"{resource.subdir}/" if resource.subdir else ""
        allow_patterns = [f"{prefix}{pattern}" for pattern in ("*.tsv", "*.jsonl", "*.json", "*.md", "*.yaml")]
    local_root = Path(
        snapshot_download(
            repo_id=resource.repo_id,
            repo_type=resource.repo_type,
            revision=revision,
            local_dir=output_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=list(resource.ignore_patterns) or None,
        )
    )
    concrete = local_root / resource.subdir if resource.subdir else local_root
    if not concrete.exists():
        raise FileNotFoundError(f"download completed but expected path is missing: {concrete}")
    return concrete.resolve()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource", help="Resource name from the config: " + ", ".join(RESOURCES))
    parser.add_argument("--config", type=Path, help="Override configs/resources.yaml.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/huggingface"))
    parser.add_argument("--repo-id", help="Override the configured Hugging Face repository ID.")
    parser.add_argument("--repo-type", choices=("dataset", "model"))
    parser.add_argument("--subdir", help="Override the repository subdirectory; pass an empty string for root.")
    parser.add_argument("--revision", help="Optional Hub branch, tag, or commit revision.")
    parser.add_argument("--metadata-only", action="store_true", help="Download manifests/configuration without media or weights.")
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan without writing files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.resource not in load_resources(args.config):
        parser.error(f"unknown resource: {args.resource}")
    resource = resolve_resource(
        args.resource,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        subdir=args.subdir,
        config=args.config,
    )
    plan = {
        "resource": args.resource,
        **asdict(resource),
        "revision": args.revision,
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "metadata_only": args.metadata_only,
    }
    if resource.status == "pre_release":
        print(
            f"warning: {args.resource} currently points to a pre-release resource; "
            "see docs/DATA.md#historical-resources",
            file=sys.stderr,
        )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not resource.repo_id:
        parser.error(
            f"{args.resource}: the final repository ID is still blank. "
            "Pass --repo-id with a confirmed repository ID, or use "
            "--config configs/resources.preview.yaml for the historical bundle."
        )
    local_path = download_resource(resource, output_dir=args.output_dir, revision=args.revision, metadata_only=args.metadata_only)
    plan["local_path"] = str(local_path)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
