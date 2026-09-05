from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

from src.workflow_a.adapters.llamacpp_server_adapter import LlamaCppServerAdapter
from src.workflow_a.pipeline import WorkflowAPipeline


SUPPORTED_MODELS = {
    "llamacpp": LlamaCppServerAdapter,
}


def run_batch(
    pipeline: WorkflowAPipeline,
    image_paths: list[Path],
    *,
    output_dir: str,
    max_new_tokens: int,
    include_audioset_nodes: bool,
    use_legacy_core: bool,
    call_mode: str,
) -> Iterator[tuple[Path, str | None]]:
    """Run the pipeline once per image, yielding (image_path, error_or_None) as
    each one finishes -- not batched into a dict returned at the end, so the
    caller can print live progress instead of going silent for the whole round."""
    for image_path in image_paths:
        print(f"Processing {image_path.name}", flush=True)
        try:
            pipeline.run(
                image_path=image_path,
                output_dir=output_dir,
                max_new_tokens=max_new_tokens,
                include_audioset_nodes=include_audioset_nodes,
                use_legacy_core=use_legacy_core,
                call_mode=call_mode,
            )
            yield image_path, None
        except Exception as exc:
            yield image_path, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Workflow A over a directory of images, and automatically retry "
            "-- only for the images that failed -- up to --max-retries times. "
            "Meant for unattended batch runs before evaluation, so a transient "
            "VLM parse/validation failure on a handful of images doesn't require "
            "manually re-running the whole batch."
        )
    )

    parser.add_argument(
        "--image",
        help="Single input image path (retries still apply).",
    )

    parser.add_argument(
        "--image-dir",
        help="Directory with input images.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--model-id",
        default=None,
    )

    parser.add_argument(
        "--server-url",
        default="http://localhost:8889",
        help="llama.cpp OpenAI-compatible server URL.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/workflow_a",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--include-audioset-nodes",
        action="store_true",
        help=(
            "Ask the VLM to add a top-level acoustic_semantics section with "
            "visually inferred AudioSet ontology nodes. "
            "Only works if --legacy-visual-core is passed!"
        ),
    )

    parser.add_argument(
        "--legacy-visual-core",
        action="store_true",
        help="Uses the legacy visual core instead of the current audioset core.",
    )

    parser.add_argument(
        "--call-mode",
        choices=["single", "three"],
        default="single",
        help="Only meaningful without --legacy-visual-core. See run_workflow_a.py --help.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help=(
            "Extra attempts per failing image after the initial run (default 2, "
            "so 3 total attempts per image before giving up on it)."
        ),
    )

    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between retry rounds (default 0).",
    )

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image-dir.")

    model_id = args.model_id or "local-vlm"

    adapter = LlamaCppServerAdapter(
        model_id=model_id,
        base_url=args.server_url,
    )

    pipeline = WorkflowAPipeline(adapter)

    if args.image_dir:
        image_paths = sorted(Path(args.image_dir).glob("*.jpg"))
        if args.limit is not None:
            image_paths = image_paths[: args.limit]
    else:
        image_paths = [Path(args.image)]

    pending = image_paths
    last_errors: dict[Path, str | None] = {}
    total_attempts = args.max_retries + 1

    for attempt in range(1, total_attempts + 1):
        if not pending:
            break

        label = "Initial run" if attempt == 1 else f"Retry {attempt - 1}/{args.max_retries}"
        print(f"{label}: {len(pending)} image(s)", flush=True)

        still_pending = []
        for image_path, error in run_batch(
            pipeline,
            pending,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            include_audioset_nodes=args.include_audioset_nodes,
            use_legacy_core=args.legacy_visual_core,
            call_mode=args.call_mode,
        ):
            if error is None:
                print(f"  OK: {image_path.name}", flush=True)
                last_errors.pop(image_path, None)
            else:
                print(f"  FAILED: {image_path.name} -> {error}", flush=True)
                last_errors[image_path] = error
                still_pending.append(image_path)

        pending = still_pending

        if pending and attempt < total_attempts and args.retry_delay > 0:
            time.sleep(args.retry_delay)

    succeeded = [p for p in image_paths if p not in last_errors]
    failed = [p for p in image_paths if p in last_errors]

    print(flush=True)
    print(f"Done: {len(succeeded)}/{len(image_paths)} succeeded, {len(failed)} permanently failed.", flush=True)
    if failed:
        print("Permanently failed images (after exhausting retries):", flush=True)
        for image_path in failed:
            print(f"  {image_path.name}: {last_errors[image_path]}", flush=True)

    report = {
        "total_images": len(image_paths),
        "succeeded": [str(p) for p in succeeded],
        "failed": [
            {"image_path": str(p), "error": last_errors[p]}
            for p in failed
        ],
        "max_retries": args.max_retries,
    }
    report_path = Path(args.output_dir) / "retry_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {report_path}", flush=True)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
