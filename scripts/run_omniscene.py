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
from concurrent.futures import ThreadPoolExecutor
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


def get_iteration_metrics(results: Dict, iteration: int) -> Optional[Dict[str, float]]:
    entry = results.get(f"ours_{iteration}")
    if not isinstance(entry, dict):
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
    return tuple(iteration for iteration in eval_iterations if get_iteration_metrics(results, iteration) is not None)


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
    completed_metrics = set(completed_iterations(model_root, eval_iterations))
    completed_timings = set(timed_iterations(model_root, eval_iterations))
    if len(completed_metrics) == len(eval_iterations) and len(completed_timings) == len(eval_iterations):
        print(f"Skipping completed scene: {scene_name}", flush=True)
        return "skipped"

    scene_dir = loader.scene_path(token)
    if not args.dry_run:
        scene_dir = loader.prepare_scene(token, force_rebuild=args.rebuild_cache)

    missing_metrics = [iteration for iteration in eval_iterations if iteration not in completed_metrics]
    missing_timings = [iteration for iteration in eval_iterations if iteration not in completed_timings]
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

            done = [future for future in future_to_job if future.done()]
            for future in done:
                gpu, scene_name = future_to_job.pop(future)
                reserved.discard(gpu)
                try:
                    status = future.result()
                    print(f"Job {scene_name}: {status}", flush=True)
                except Exception as exc:
                    failures.append({"scene": scene_name, "error": str(exc)})
                    print(f"Job {scene_name} failed: {exc}", flush=True)
            if (jobs or future_to_job) and not args.dry_run:
                time.sleep(5)
    return failures


def write_summary(
    scene_names: Sequence[str],
    eval_iterations: Sequence[int],
    failures: Sequence[Dict[str, str]],
    args: argparse.Namespace,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_iteration = {}
    incomplete_scenes = []

    for iteration in eval_iterations:
        values = {name: [] for name in METRIC_NAMES}
        training_seconds = []
        missing_metrics = []
        missing_timings = []
        for scene_name in scene_names:
            model_root = args.output_dir / scene_name
            metrics = get_iteration_metrics(load_results(model_root), iteration)
            if metrics is None:
                missing_metrics.append(scene_name)
            else:
                for name in METRIC_NAMES:
                    values[name].append(metrics[name])
            elapsed_seconds = get_iteration_training_seconds(load_training_times(model_root), iteration)
            if elapsed_seconds is None:
                missing_timings.append(scene_name)
            else:
                training_seconds.append(elapsed_seconds)
        per_iteration[str(iteration)] = {
            "num_scenes": len(scene_names) - len(missing_metrics),
            "num_timed_scenes": len(scene_names) - len(missing_timings),
            "mean": {
                name: (sum(values[name]) / len(values[name]) if values[name] else None)
                for name in METRIC_NAMES
            },
            "mean_training_seconds": (
                sum(training_seconds) / len(training_seconds) if training_seconds else None
            ),
            "missing_metric_scenes": missing_metrics,
            "missing_timing_scenes": missing_timings,
        }

    for scene_name in scene_names:
        model_root = args.output_dir / scene_name
        completed_metrics = set(completed_iterations(model_root, eval_iterations))
        completed_timings = set(timed_iterations(model_root, eval_iterations))
        missing_metrics = [iteration for iteration in eval_iterations if iteration not in completed_metrics]
        missing_timings = [iteration for iteration in eval_iterations if iteration not in completed_timings]
        if missing_metrics or missing_timings:
            incomplete_scenes.append(
                {
                    "scene": scene_name,
                    "missing_metric_iterations": missing_metrics,
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
            ["iteration", "num_scenes", "num_timed_scenes", "mean_training_seconds", *METRIC_NAMES]
        )
        for iteration in eval_iterations:
            entry = per_iteration[str(iteration)]
            writer.writerow(
                [
                    iteration,
                    entry["num_scenes"],
                    entry["num_timed_scenes"],
                    entry["mean_training_seconds"],
                    *[entry["mean"][name] for name in METRIC_NAMES],
                ]
            )

    print("\nAggregated results:")
    print("iteration  scenes  time(s)      PSNR       SSIM       LPIPS")
    for iteration in eval_iterations:
        entry = per_iteration[str(iteration)]
        means = entry["mean"]
        mean_time = entry["mean_training_seconds"]
        formatted_time = "N/A" if mean_time is None else f"{mean_time:.3f}"
        formatted = ["N/A" if means[name] is None else f"{means[name]:.7f}" for name in METRIC_NAMES]
        print(
            f"{iteration:>9}  {entry['num_scenes']:>6}  {formatted_time:>10}  "
            f"{formatted[0]:>9}  {formatted[1]:>9}  {formatted[2]:>9}"
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
