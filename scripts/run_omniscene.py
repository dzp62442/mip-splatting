import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import GPUtil

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from comp_svfgs.omniscene_dataset import LoaderConfig, OmniSceneLoader


def parse_resolution(value: str):
    if "x" not in value:
        raise argparse.ArgumentTypeError("Resolution must be formatted as HxW, e.g., 112x200.")
    h, w = value.lower().split("x")
    return int(h), int(w)


def parse_args():
    parser = argparse.ArgumentParser(description="Run mip-splatting pipeline on OmniScene.")
    parser.add_argument("--data_root", type=str, default="datasets/omniscene", help="Path to OmniScene dataset root.")
    parser.add_argument("--cache_root", type=str, default="output", help="Root folder to store cached Blender scenes.")
    parser.add_argument("--stage", type=str, default="val", choices=["train", "val", "test", "demo"])
    parser.add_argument("--resolution", type=parse_resolution, default="112x200", help="Target resolution HxW.")
    parser.add_argument("--output_dir", type=str, default="output/omniscene_runs", help="Directory to store training outputs.")
    parser.add_argument("--kernel_size", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=10_000, help="Number of training iterations.")
    parser.add_argument("--gpus", type=str, default=None, help="Comma separated GPU ids. Defaults to auto detection.")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    return parser.parse_args()


def run_command(cmd: str, dry_run: bool):
    print(cmd)
    if not dry_run:
        os.system(cmd)


def train_scene(gpu, scene_token, scene_dir, args):
    model_root = os.path.join(args.output_dir, scene_token)
    os.makedirs(args.output_dir, exist_ok=True)

    base_env = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
    train_cmd = (
        f"{base_env}python train.py "
        f"-s {scene_dir} -m {model_root} --eval --port {6009 + int(gpu)} "
        f"--iterations {args.iterations} "
        f"--kernel_size {args.kernel_size} -r 1"
    )
    render_cmd = f"{base_env}python render.py -m {model_root} --skip_train"
    metrics_cmd = f"{base_env}python metrics.py -m {model_root} -r 1"

    run_command(train_cmd, args.dry_run)
    run_command(render_cmd, args.dry_run)
    run_command(metrics_cmd, args.dry_run)


def worker(gpu, job, args):
    scene_token, scene_dir = job
    print(f"Starting {scene_token} on GPU {gpu}")
    train_scene(gpu, scene_token, scene_dir, args)
    print(f"Finished {scene_token} on GPU {gpu}")


def get_available_gpus(excluded):
    available = set(GPUtil.getAvailable(order="first", limit=16, maxMemory=0.1))
    return list(available - excluded)


def dispatch_jobs(jobs, args):
    future_to_job = {}
    reserved = set()
    excluded = set()

    gpus = None
    if args.gpus is not None:
        gpus = {int(x) for x in args.gpus.split(",")}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        while jobs or future_to_job:
            if gpus is not None:
                available_gpus = list(gpus - reserved - excluded)
            else:
                available_gpus = get_available_gpus(reserved | excluded)

            while available_gpus and jobs:
                gpu = available_gpus.pop(0)
                job = jobs.pop(0)
                future = executor.submit(worker, gpu, job, args)
                future_to_job[future] = gpu
                reserved.add(gpu)

            done = [f for f in future_to_job if f.done()]
            for future in done:
                gpu = future_to_job.pop(future)
                reserved.discard(gpu)
                try:
                    future.result()
                except Exception as exc:
                    print(f"Job failed on GPU {gpu}: {exc}")
            time.sleep(5)


def main():
    args = parse_args()
    cfg = LoaderConfig(
        data_root=Path(args.data_root),
        cache_root=Path(args.cache_root),
        stage=args.stage,
        resolution=args.resolution,
    )
    loader = OmniSceneLoader(cfg)

    jobs = []
    tokens = list(loader.list_tokens())
    for idx, token in enumerate(tokens):
        scene_dir = loader.prepare_scene(token, force_rebuild=args.rebuild_cache)
        scene_name = f"{idx+1:02d}_{token}"
        jobs.append((scene_name, str(scene_dir)))

    dispatch_jobs(jobs, args)


if __name__ == "__main__":
    main()
