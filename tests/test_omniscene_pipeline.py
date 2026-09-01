import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comp_svfgs.omniscene_dataset import LoaderConfig, OmniSceneLoader
from scripts import run_omniscene


class OmniScenePipelineTest(unittest.TestCase):
    def test_center150_loader_and_scene_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            version_root = root / "data" / "interp_12Hz_trainval"
            version_root.mkdir(parents=True)
            tokens = [f"scene{i:032x}_bin000" for i in range(150)]
            (version_root / "bins_center150_v1.json").write_text(json.dumps({"bins": tokens}))

            loader = OmniSceneLoader(
                LoaderConfig(data_root=root / "data", cache_root=root / "cache", stage="center150")
            )

            self.assertEqual(list(loader.list_tokens()), tokens)
            self.assertTrue(loader.scene_name(tokens[0]).startswith("001_"))
            self.assertTrue(loader.scene_name(tokens[-1]).startswith("150_"))

    def test_resume_uses_latest_checkpoint_before_missing_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = root / "model"
            model_root.mkdir()
            (model_root / "chkpnt1000.pth").write_bytes(b"checkpoint")
            (model_root / "chkpnt5000.pth").write_bytes(b"checkpoint")
            args = argparse.Namespace(iterations=10_000, kernel_size=0.1, dry_run=True)

            with mock.patch.object(run_omniscene, "run_command") as run_command:
                run_omniscene.train_if_needed(
                    0,
                    root / "scene",
                    model_root,
                    [5_000, 10_000],
                    [],
                    run_omniscene.CENTER150_EVAL_ITERATIONS,
                    args,
                )

            command = run_command.call_args.args[0]
            checkpoint_index = command.index("--start_checkpoint") + 1
            self.assertEqual(Path(command[checkpoint_index]).name, "chkpnt1000.pth")
            test_iteration_index = command.index("--test_iterations") + 1
            self.assertEqual(command[test_iteration_index], "10001")
            timing_index = command.index("--timing_iterations") + 1
            self.assertEqual(command[timing_index:timing_index + 3], ["1000", "5000", "10000"])

    def test_summary_averages_scenes_at_each_iteration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            scene_names = ["001_scene", "002_scene"]
            for scene_index, scene_name in enumerate(scene_names):
                scene_dir = output_dir / scene_name
                scene_dir.mkdir()
                results = {}
                for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS:
                    results[f"ours_{iteration}"] = {
                        "PSNR": 10.0 + scene_index * 4.0 + iteration / 1000.0,
                        "SSIM": 0.5 + scene_index * 0.2,
                        "LPIPS": 0.3 - scene_index * 0.1,
                    }
                (scene_dir / "results.json").write_text(json.dumps(results))
                (scene_dir / "training_times.json").write_text(
                    json.dumps(
                        {
                            "elapsed_seconds": {
                                "1000": 10.0 + scene_index * 2.0,
                                "5000": 50.0 + scene_index * 4.0,
                                "10000": 100.0 + scene_index * 6.0,
                            }
                        }
                    )
                )

            args = argparse.Namespace(output_dir=output_dir, stage="center150")
            summary_path = run_omniscene.write_summary(
                scene_names, run_omniscene.CENTER150_EVAL_ITERATIONS, [], args
            )
            summary = json.loads(summary_path.read_text())

            self.assertTrue(summary["complete"])
            self.assertEqual(summary["completed_scenes"], 2)
            self.assertAlmostEqual(summary["metrics"]["1000"]["mean"]["PSNR"], 13.0)
            self.assertAlmostEqual(summary["metrics"]["5000"]["mean"]["SSIM"], 0.6)
            self.assertAlmostEqual(summary["metrics"]["10000"]["mean"]["LPIPS"], 0.25)
            self.assertAlmostEqual(summary["metrics"]["1000"]["mean_training_seconds"], 11.0)
            self.assertAlmostEqual(summary["metrics"]["10000"]["mean_training_seconds"], 103.0)

    def test_incomplete_metrics_do_not_mark_scene_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir)
            (model_root / "results.json").write_text(
                json.dumps({"ours_1000": {"PSNR": 20.0, "SSIM": 0.8}})
            )
            self.assertEqual(
                run_omniscene.completed_iterations(
                    model_root, run_omniscene.CENTER150_EVAL_ITERATIONS
                ),
                (),
            )

    def test_summary_requires_training_times(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            scene_dir = output_dir / "001_scene"
            scene_dir.mkdir()
            results = {
                f"ours_{iteration}": {"PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.2}
                for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS
            }
            (scene_dir / "results.json").write_text(json.dumps(results))
            args = argparse.Namespace(output_dir=output_dir, stage="center150")

            summary_path = run_omniscene.write_summary(
                ["001_scene"], run_omniscene.CENTER150_EVAL_ITERATIONS, [], args
            )
            summary = json.loads(summary_path.read_text())

            self.assertFalse(summary["complete"])
            self.assertEqual(summary["completed_scenes"], 0)
            self.assertEqual(
                summary["incomplete_scenes"][0]["missing_timing_iterations"],
                [1000, 5000, 10000],
            )


if __name__ == "__main__":
    unittest.main()
