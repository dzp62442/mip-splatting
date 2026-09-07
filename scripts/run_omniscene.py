import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import GPUtil

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from comp_svfgs.omniscene_dataset import LoaderConfig, OmniSceneLoader


CENTER150_EVAL_ITERATIONS = (1_000, 5_000, 10_000)
METRIC_NAMES = ("PSNR", "SSIM", "LPIPS")
EXPECTED_TEST_VIEWS = 18
NOVEL_TEST_VIEWS = 12
ALL_VIEWS_SCOPE = "all_18_views"
NOVEL_VIEWS_SCOPE = "novel_12_views"
METRIC_SCOPE_VIEW_COUNTS = {
    ALL_VIEWS_SCOPE: EXPECTED_TEST_VIEWS,
    NOVEL_VIEWS_SCOPE: NOVEL_TEST_VIEWS,
}
CHECKPOINT_PATTERN = re.compile(r"^chkpnt(\d+)\.pth$")
TRAINING_TIMES_FILENAME = "training_times.json"


def parse_resolution(value: str) -> Tuple[int, int]:
    if "x" not in value:
        raise argparse.ArgumentTypeError("Resolution must be formatted as HxW, e.g., 112x200.")
    try:
        h, w = value.lower().split("x")
        resolution = int(h), int(w)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution must be formatted as HxW, e.g., 112x200.") from exc
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise argparse.ArgumentTypeError("Resolution dimensions must be positive.")
    return resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mip-splatting pipeline on OmniScene.")
    parser.add_argument("--data_root", type=str, default="datasets/omniscene", help="Path to OmniScene dataset root.")
    parser.add_argument("--cache_root", type=str, default="output", help="Root folder to store cached Blender scenes.")
    parser.add_argument("--stage", type=str, default="center150", choices=["train", "val", "center150", "test", "demo"])
    parser.add_argument("--resolution", type=parse_resolution, default="112x200", help="Target resolution HxW.")
    parser.add_argument("--output_dir", type=str, default="output/omniscene_runs", help="Directory to store training outputs.")
    parser.add_argument("--kernel_size", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=10_000, help="Number of training iterations.")
    parser.add_argument(
        "--eval_iterations",
        nargs="+",
        type=int,
        default=None,
        help="Iterations to render and evaluate. Defaults to 1k/5k/10k for center150 and the final iteration otherwise.",
    )
    parser.add_argument("--gpus", type=str, default=None, help="Comma separated GPU ids. Defaults to auto detection.")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    return parser.parse_args()


def resolve_from_root(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved.resolve()


def resolve_eval_iterations(args: argparse.Namespace) -> Tuple[int, ...]:
    if args.eval_iterations is None:
        iterations = CENTER150_EVAL_ITERATIONS if args.stage == "center150" else (args.iterations,)
    else:
        iterations = tuple(args.eval_iterations)
    iterations = tuple(sorted(set(iterations)))
    if not iterations or iterations[0] <= 0:
        raise ValueError("Evaluation iterations must be positive.")
    if iterations[-1] > args.iterations:
        raise ValueError(
            f"Evaluation iteration {iterations[-1]} exceeds total training iterations {args.iterations}."
        )
    if args.stage == "center150" and iterations != CENTER150_EVAL_ITERATIONS:
        raise ValueError(
            "center150 uses the fixed evaluation milestones 1000, 5000, and 10000. "
            "Omit --eval_iterations or pass exactly those values."
        )
    return iterations


def format_command(command: Sequence[str], env: Dict[str, str]) -> str:
    env_prefix = " ".join(
        f"{key}={shlex.quote(env[key])}" for key in ("OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
    )
    return f"{env_prefix} {' '.join(shlex.quote(str(arg)) for arg in command)}"


def run_command(command: Sequence[str], env: Dict[str, str], dry_run: bool) -> None:
    print(format_command(command, env), flush=True)
    if not dry_run:
        subprocess.run(list(command), cwd=str(ROOT_DIR), env=env, check=True)


def load_results(model_root: Path) -> Dict:
    results_path = model_root / "results.json"
    if not results_path.is_file():
        return {}
    try:
        data = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def get_iteration_metrics(
    results: Dict,
    iteration: int,
    scope: str = ALL_VIEWS_SCOPE,
) -> Optional[Dict[str, float]]:
    entry = results.get(f"ours_{iteration}")
    if not isinstance(entry, dict):
        return None
    scoped_entry = entry.get(scope)
    if isinstance(scoped_entry, dict):
        if scoped_entry.get("num_views") != METRIC_SCOPE_VIEW_COUNTS[scope]:
            return None
        entry = scoped_entry
    elif scope != ALL_VIEWS_SCOPE:
        return None
    metrics = {}
    for name in METRIC_NAMES:
        try:
            value = float(entry[name])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        metrics[name] = value
    return metrics


def completed_iterations(model_root: Path, eval_iterations: Sequence[int]) -> Tuple[int, ...]:
    results = load_results(model_root)
    return tuple(
        iteration
        for iteration in eval_iterations
        if all(
            get_iteration_metrics(results, iteration, scope) is not None
            for scope in METRIC_SCOPE_VIEW_COUNTS
        )
    )


def load_per_view_results(model_root: Path) -> Dict:
    per_view_path = model_root / "per_view.json"
    if not per_view_path.is_file():
        return {}
    try:
        data = json.loads(per_view_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def get_per_view_metrics(
    per_view_results: Dict,
    iteration: int,
    view_count: int,
) -> Optional[Dict[str, float]]:
    entry = per_view_results.get(f"ours_{iteration}")
    if not isinstance(entry, dict):
        return None

    metric_values = {}
    image_names = None
    for name in METRIC_NAMES:
        values = entry.get(name)
        if not isinstance(values, dict):
            return None
        current_names = set(values)
        if image_names is None:
            image_names = current_names
        elif current_names != image_names:
            return None
        metric_values[name] = values

    if image_names is None or len(image_names) != EXPECTED_TEST_VIEWS:
        return None
    selected_names = sorted(image_names)[:view_count]
    metrics = {}
    for name in METRIC_NAMES:
        try:
            values = [float(metric_values[name][image_name]) for image_name in selected_names]
        except (KeyError, TypeError, ValueError):
            return None
        if len(values) != view_count or not all(math.isfinite(value) for value in values):
            return None
        metrics[name] = sum(values) / len(values)
    return metrics


def ensure_metric_scopes(model_root: Path, eval_iterations: Sequence[int]) -> bool:
    """Add all-view and novel-view means to existing metric files without reevaluation."""
    results = load_results(model_root)
    per_view_results = load_per_view_results(model_root)
    if not results or not per_view_results:
        return False

    changed = False
    for iteration in eval_iterations:
        key = f"ours_{iteration}"
        entry = results.get(key)
        if not isinstance(entry, dict):
            continue

        all_view_metrics = get_iteration_metrics(results, iteration, ALL_VIEWS_SCOPE)
        per_view_all_metrics = get_per_view_metrics(
            per_view_results, iteration, EXPECTED_TEST_VIEWS
        )
        novel_view_metrics = get_per_view_metrics(
            per_view_results, iteration, NOVEL_TEST_VIEWS
        )
        if per_view_all_metrics is None or novel_view_metrics is None:
            continue
        if all_view_metrics is None:
            all_view_metrics = per_view_all_metrics
            entry.update(all_view_metrics)

        scoped_metrics = {
            ALL_VIEWS_SCOPE: {
                "num_views": EXPECTED_TEST_VIEWS,
                **all_view_metrics,
            },
            NOVEL_VIEWS_SCOPE: {
                "num_views": NOVEL_TEST_VIEWS,
                **novel_view_metrics,
            },
        }
        for scope, values in scoped_metrics.items():
            if entry.get(scope) != values:
                entry[scope] = values
                changed = True

    if changed:
        results_path = model_root / "results.json"
        temp_path = results_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(results, indent=2) + "\n")
        temp_path.replace(results_path)
    return changed


def load_training_times(model_root: Path) -> Dict:
    timing_path = model_root / TRAINING_TIMES_FILENAME
    if not timing_path.is_file():
        return {}
    try:
        data = json.loads(timing_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def get_iteration_training_seconds(timing: Dict, iteration: int) -> Optional[float]:
    try:
        value = float(timing["elapsed_seconds"][str(iteration)])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def timed_iterations(model_root: Path, eval_iterations: Sequence[int]) -> Tuple[int, ...]:
    timing = load_training_times(model_root)
    return tuple(
        iteration
        for iteration in eval_iterations
        if get_iteration_training_seconds(timing, iteration) is not None
    )


def point_cloud_path(model_root: Path, iteration: int) -> Path:
    return model_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"


def point_cloud_exists(model_root: Path, iteration: int) -> bool:
    path = point_cloud_path(model_root, iteration)
    return path.is_file() and path.stat().st_size > 0


def latest_checkpoint_before(model_root: Path, iteration: int) -> Optional[Path]:
    candidates = []
    for path in model_root.glob("chkpnt*.pth"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file() or path.stat().st_size == 0:
            continue
        checkpoint_iteration = int(match.group(1))
        if checkpoint_iteration < iteration:
            candidates.append((checkpoint_iteration, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def render_is_complete(model_root: Path, iteration: int, resolution_scale: int = 1) -> bool:
    method_dir = model_root / "test" / f"ours_{iteration}"
    render_dir = method_dir / f"test_preds_{resolution_scale}"
    gt_dir = method_dir / f"gt_{resolution_scale}"
    render_names = {path.name for path in render_dir.glob("*.png")}
    gt_names = {path.name for path in gt_dir.glob("*.png")}
    return len(render_names) == EXPECTED_TEST_VIEWS and render_names == gt_names


def build_environment(gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def train_if_needed(
    gpu: int,
    scene_dir: Path,
    model_root: Path,
    missing_metric_iterations: Sequence[int],
    missing_timing_iterations: Sequence[int],
    eval_iterations: Sequence[int],
    args: argparse.Namespace,
) -> None:
    missing_point_clouds = [
        iteration
        for iteration in missing_metric_iterations
        if not point_cloud_exists(model_root, iteration)
    ]
    if not missing_point_clouds and not missing_timing_iterations:
        print(f"Training artifacts already available: {model_root.name}", flush=True)
        return

    first_missing = min([*missing_point_clouds, *missing_timing_iterations])
    checkpoint = latest_checkpoint_before(model_root, first_missing)
    save_iterations = tuple(sorted(set(eval_iterations) | {args.iterations}))
    env = build_environment(gpu)
    command = [
        sys.executable,
        "train.py",
        "-s",
        str(scene_dir),
        "-m",
        str(model_root),
        "--eval",
        "--port",
        str(6009 + gpu),
        "--iterations",
        str(args.iterations),
        "--kernel_size",
        str(args.kernel_size),
        "-r",
        "1",
        "--test_iterations",
        str(args.iterations + 1),
        "--save_iterations",
        *[str(iteration) for iteration in save_iterations],
        "--checkpoint_iterations",
        *[str(iteration) for iteration in save_iterations],
        "--timing_iterations",
        *[str(iteration) for iteration in eval_iterations],
    ]
    if checkpoint is not None:
        command.extend(["--start_checkpoint", str(checkpoint)])
        print(f"Resuming {model_root.name} from {checkpoint.name}", flush=True)
    else:
        print(f"Starting {model_root.name} from scratch", flush=True)
    run_command(command, env, args.dry_run)

    if args.dry_run:
        return
    still_missing_point_clouds = [
        iteration
        for iteration in missing_metric_iterations
        if not point_cloud_exists(model_root, iteration)
    ]
    still_missing_timings = [
        iteration
        for iteration in missing_timing_iterations
        if iteration not in set(timed_iterations(model_root, eval_iterations))
    ]
    if still_missing_point_clouds or still_missing_timings:
        raise RuntimeError(
            "Training finished with incomplete artifacts: "
            f"point_clouds={still_missing_point_clouds}, timings={still_missing_timings}, model={model_root}"
        )


def evaluate_missing_iterations(
    gpu: int,
    model_root: Path,
    eval_iterations: Sequence[int],
    args: argparse.Namespace,
) -> None:
    ensure_metric_scopes(model_root, eval_iterations)
    completed = set(completed_iterations(model_root, eval_iterations))
    missing = [iteration for iteration in eval_iterations if iteration not in completed]
    if not missing:
        return

    env = build_environment(gpu)
    for iteration in missing:
        if render_is_complete(model_root, iteration):
            print(f"Render already available: {model_root.name} @ {iteration}", flush=True)
            continue
        command = [
            sys.executable,
            "render.py",
            "-m",
            str(model_root),
            "--iteration",
            str(iteration),
            "--data_device",
            "cpu",
            "--skip_train",
        ]
        run_command(command, env, args.dry_run)

    command = [sys.executable, "metrics.py", "-m", str(model_root), "-r", "1"]
    run_command(command, env, args.dry_run)
    if args.dry_run:
        return
    ensure_metric_scopes(model_root, eval_iterations)

    remaining = [
        iteration
        for iteration in eval_iterations
        if iteration not in set(completed_iterations(model_root, eval_iterations))
    ]
    if remaining:
        raise RuntimeError(f"Metrics are missing at iterations {remaining}: {model_root}")


def process_scene(
    gpu: int,
    token: str,
    scene_name: str,
    loader: OmniSceneLoader,
    eval_iterations: Sequence[int],
    args: argparse.Namespace,
) -> str:
    model_root = args.output_dir / scene_name
    metrics_updated = ensure_metric_scopes(model_root, eval_iterations)
    completed_metrics = set(completed_iterations(model_root, eval_iterations))
    completed_timings = set(timed_iterations(model_root, eval_iterations))
    if len(completed_metrics) == len(eval_iterations) and len(completed_timings) == len(eval_iterations):
        if metrics_updated:
            print(f"Skipping completed experiment; metric scopes updated: {scene_name}", flush=True)
            return "metrics-updated"
        print(f"Skipping completed scene: {scene_name}", flush=True)
        return "skipped"

    scene_dir = loader.scene_path(token)
    missing_metrics = [iteration for iteration in eval_iterations if iteration not in completed_metrics]
    missing_timings = [iteration for iteration in eval_iterations if iteration not in completed_timings]
    needs_training = bool(missing_timings) or any(
        not point_cloud_exists(model_root, iteration) for iteration in missing_metrics
    )
    if needs_training and not args.dry_run:
        scene_dir = loader.prepare_scene(token, force_rebuild=args.rebuild_cache)

    train_if_needed(
        gpu, scene_dir, model_root, missing_metrics, missing_timings, eval_iterations, args
    )
    evaluate_missing_iterations(gpu, model_root, eval_iterations, args)
    if len(set(timed_iterations(model_root, eval_iterations))) != len(eval_iterations):
        raise RuntimeError(f"Training times are incomplete: {model_root}")
    print(f"Completed scene: {scene_name}", flush=True)
    return "completed"


def get_available_gpus(excluded: Sequence[int]) -> List[int]:
    available = set(GPUtil.getAvailable(order="first", limit=16, maxMemory=0.1))
    return sorted(available - set(excluded))


def dispatch_jobs(
    jobs: List[Tuple[str, str]],
    loader: OmniSceneLoader,
    eval_iterations: Sequence[int],
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    future_to_job = {}
    reserved = set()
    failures = []

    selected_gpus = None
    if args.gpus is not None:
        selected_gpus = {int(value.strip()) for value in args.gpus.split(",") if value.strip()}
        if not selected_gpus:
            raise ValueError("--gpus must contain at least one GPU id.")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        while jobs or future_to_job:
            if args.dry_run and selected_gpus is None:
                available_gpus = [0] if 0 not in reserved else []
            elif selected_gpus is not None:
                available_gpus = sorted(selected_gpus - reserved)
            else:
                available_gpus = get_available_gpus(reserved)

            while available_gpus and jobs and len(future_to_job) < args.max_workers:
                gpu = available_gpus.pop(0)
                token, scene_name = jobs.pop(0)
                print(f"Starting {scene_name} on GPU {gpu}", flush=True)
                future = executor.submit(
                    process_scene, gpu, token, scene_name, loader, eval_iterations, args
                )
                future_to_job[future] = (gpu, scene_name)
                reserved.add(gpu)

            if future_to_job:
                done, _ = wait(
                    tuple(future_to_job),
                    timeout=5,
                    return_when=FIRST_COMPLETED,
                )
            else:
                done = ()
            for future in done:
                gpu, scene_name = future_to_job.pop(future)
                reserved.discard(gpu)
                try:
                    status = future.result()
                    print(f"Job {scene_name}: {status}", flush=True)
                except Exception as exc:
                    failures.append({"scene": scene_name, "error": str(exc)})
                    print(f"Job {scene_name} failed: {exc}", flush=True)
            if jobs and not future_to_job and not done and not args.dry_run:
                time.sleep(5)
    return failures


def write_summary(
    scene_names: Sequence[str],
    eval_iterations: Sequence[int],
    failures: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scene_name in scene_names:
        ensure_metric_scopes(args.output_dir / scene_name, eval_iterations)
    per_iteration = {}
    incomplete_scenes = []

    for iteration in eval_iterations:
        scoped_values = {
            scope: {name: [] for name in METRIC_NAMES}
            for scope in METRIC_SCOPE_VIEW_COUNTS
        }
        scoped_missing_metrics = {scope: [] for scope in METRIC_SCOPE_VIEW_COUNTS}
        training_seconds = []
        missing_timings = []
        for scene_name in scene_names:
            model_root = args.output_dir / scene_name
            results = load_results(model_root)
            for scope in METRIC_SCOPE_VIEW_COUNTS:
                metrics = get_iteration_metrics(results, iteration, scope)
                if metrics is None:
                    scoped_missing_metrics[scope].append(scene_name)
                else:
                    for name in METRIC_NAMES:
                        scoped_values[scope][name].append(metrics[name])
            elapsed_seconds = get_iteration_training_seconds(load_training_times(model_root), iteration)
            if elapsed_seconds is None:
                missing_timings.append(scene_name)
            else:
                training_seconds.append(elapsed_seconds)
        metric_scopes = {}
        for scope, view_count in METRIC_SCOPE_VIEW_COUNTS.items():
            values = scoped_values[scope]
            missing_metrics = scoped_missing_metrics[scope]
            metric_scopes[scope] = {
                "num_views_per_scene": view_count,
                "num_scenes": len(scene_names) - len(missing_metrics),
                "mean": {
                    name: (sum(values[name]) / len(values[name]) if values[name] else None)
                    for name in METRIC_NAMES
                },
                "missing_metric_scenes": missing_metrics,
            }
        all_views = metric_scopes[ALL_VIEWS_SCOPE]
        per_iteration[str(iteration)] = {
            # Backward-compatible aliases: these always refer to all 18 test views.
            "num_scenes": all_views["num_scenes"],
            "num_timed_scenes": len(scene_names) - len(missing_timings),
            "mean": all_views["mean"],
            "mean_training_seconds": (
                sum(training_seconds) / len(training_seconds) if training_seconds else None
            ),
            "missing_metric_scenes": all_views["missing_metric_scenes"],
            "missing_timing_scenes": missing_timings,
            "metric_scopes": metric_scopes,
        }

    for scene_name in scene_names:
        model_root = args.output_dir / scene_name
        completed_metrics = set(completed_iterations(model_root, eval_iterations))
        completed_timings = set(timed_iterations(model_root, eval_iterations))
        missing_metrics = [iteration for iteration in eval_iterations if iteration not in completed_metrics]
        missing_timings = [iteration for iteration in eval_iterations if iteration not in completed_timings]
        if missing_metrics or missing_timings:
            results = load_results(model_root)
            missing_metrics_by_scope = {
                scope: [
                    iteration
                    for iteration in eval_iterations
                    if get_iteration_metrics(results, iteration, scope) is None
                ]
                for scope in METRIC_SCOPE_VIEW_COUNTS
            }
            incomplete_scenes.append(
                {
                    "scene": scene_name,
                    "missing_metric_iterations": missing_metrics,
                    "missing_metric_iterations_by_scope": missing_metrics_by_scope,
                    "missing_timing_iterations": missing_timings,
                }
            )

    summary = {
        "stage": args.stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_scenes": len(scene_names),
        "completed_scenes": len(scene_names) - len(incomplete_scenes),
        "complete": not incomplete_scenes,
        "eval_iterations": list(eval_iterations),
        "metric_scope_view_counts": METRIC_SCOPE_VIEW_COUNTS,
        "metrics": per_iteration,
        "incomplete_scenes": incomplete_scenes,
        "job_failures": list(failures),
    }
    summary_path = args.output_dir / f"{args.stage}_summary.json"
    temp_path = summary_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(summary, indent=2) + "\n")
    temp_path.replace(summary_path)

    csv_path = args.output_dir / f"{args.stage}_summary.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "iteration",
                f"num_scenes_{ALL_VIEWS_SCOPE}",
                f"num_scenes_{NOVEL_VIEWS_SCOPE}",
                "num_timed_scenes",
                "mean_training_seconds",
                *[f"{ALL_VIEWS_SCOPE}_{name}" for name in METRIC_NAMES],
                *[f"{NOVEL_VIEWS_SCOPE}_{name}" for name in METRIC_NAMES],
            ]
        )
        for iteration in eval_iterations:
            entry = per_iteration[str(iteration)]
            writer.writerow(
                [
                    iteration,
                    entry["metric_scopes"][ALL_VIEWS_SCOPE]["num_scenes"],
                    entry["metric_scopes"][NOVEL_VIEWS_SCOPE]["num_scenes"],
                    entry["num_timed_scenes"],
                    entry["mean_training_seconds"],
                    *[
                        entry["metric_scopes"][ALL_VIEWS_SCOPE]["mean"][name]
                        for name in METRIC_NAMES
                    ],
                    *[
                        entry["metric_scopes"][NOVEL_VIEWS_SCOPE]["mean"][name]
                        for name in METRIC_NAMES
                    ],
                ]
            )

    print("\nAggregated results:")
    print("iteration  scope             scenes  time(s)      PSNR       SSIM       LPIPS")
    for iteration in eval_iterations:
        entry = per_iteration[str(iteration)]
        mean_time = entry["mean_training_seconds"]
        formatted_time = "N/A" if mean_time is None else f"{mean_time:.3f}"
        for scope in METRIC_SCOPE_VIEW_COUNTS:
            scope_entry = entry["metric_scopes"][scope]
            means = scope_entry["mean"]
            formatted = [
                "N/A" if means[name] is None else f"{means[name]:.7f}"
                for name in METRIC_NAMES
            ]
            print(
                f"{iteration:>9}  {scope:<17}  {scope_entry['num_scenes']:>6}  "
                f"{formatted_time:>10}  {formatted[0]:>9}  {formatted[1]:>9}  "
                f"{formatted[2]:>9}"
            )
    print(f"Summary: {summary_path}")
    return summary_path


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")
    if args.max_workers <= 0:
        raise ValueError("--max_workers must be positive.")
    eval_iterations = resolve_eval_iterations(args)

    args.data_root = resolve_from_root(args.data_root)
    args.cache_root = resolve_from_root(args.cache_root)
    args.output_dir = resolve_from_root(args.output_dir)
    cfg = LoaderConfig(
        data_root=args.data_root,
        cache_root=args.cache_root,
        stage=args.stage,
        resolution=args.resolution,
    )
    loader = OmniSceneLoader(cfg)

    tokens = list(loader.list_tokens())
    jobs = [(token, loader.scene_name(token)) for token in tokens]
    failures = dispatch_jobs(jobs.copy(), loader, eval_iterations, args)
    if args.dry_run:
        return
    write_summary([scene_name for _, scene_name in jobs], eval_iterations, failures, args)
    if failures:
        raise RuntimeError(f"{len(failures)} scene jobs failed; see the summary for details.")


if __name__ == "__main__":
    main()
