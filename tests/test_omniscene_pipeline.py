import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comp_svfgs.omniscene_dataset import LoaderConfig, OmniSceneLoader
from scripts import run_omniscene


class OmniScenePipelineTest(unittest.TestCase):
    @staticmethod
    def _metric_entry(all_metrics=None, novel_metrics=None):
        all_metrics = all_metrics or {"PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.2}
        novel_metrics = novel_metrics or {"PSNR": 18.0, "SSIM": 0.7, "LPIPS": 0.3}
        return {
            **all_metrics,
            run_omniscene.ALL_VIEWS_SCOPE: {
                "num_views": run_omniscene.EXPECTED_TEST_VIEWS,
                **all_metrics,
            },
            run_omniscene.NOVEL_VIEWS_SCOPE: {
                "num_views": run_omniscene.NOVEL_TEST_VIEWS,
                **novel_metrics,
            },
        }

    @staticmethod
    def _write_completed_scene(model_root):
        model_root.mkdir()
        results = {
            f"ours_{iteration}": OmniScenePipelineTest._metric_entry()
            for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS
        }
        (model_root / "results.json").write_text(json.dumps(results))
        (model_root / "training_times.json").write_text(
            json.dumps(
                {
                    "elapsed_seconds": {
                        str(iteration): float(iteration)
                        for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS
                    }
                }
            )
        )

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
                    all_metrics = {
                        "PSNR": 10.0 + scene_index * 4.0 + iteration / 1000.0,
                        "SSIM": 0.5 + scene_index * 0.2,
                        "LPIPS": 0.3 - scene_index * 0.1,
                    }
                    novel_metrics = {
                        "PSNR": 8.0 + scene_index * 2.0 + iteration / 1000.0,
                        "SSIM": 0.4 + scene_index * 0.1,
                        "LPIPS": 0.4 - scene_index * 0.1,
                    }
                    results[f"ours_{iteration}"] = self._metric_entry(
                        all_metrics, novel_metrics
                    )
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
            novel_summary = summary["metrics"]["1000"]["metric_scopes"][
                run_omniscene.NOVEL_VIEWS_SCOPE
            ]
            self.assertEqual(novel_summary["num_views_per_scene"], 12)
            self.assertAlmostEqual(novel_summary["mean"]["PSNR"], 10.0)
            self.assertAlmostEqual(novel_summary["mean"]["SSIM"], 0.45)
            self.assertAlmostEqual(novel_summary["mean"]["LPIPS"], 0.35)
            self.assertAlmostEqual(summary["metrics"]["1000"]["mean_training_seconds"], 11.0)
            self.assertAlmostEqual(summary["metrics"]["10000"]["mean_training_seconds"], 103.0)

            csv_header = (output_dir / "center150_summary.csv").read_text().splitlines()[0]
            self.assertIn("all_18_views_PSNR", csv_header)
            self.assertIn("novel_12_views_PSNR", csv_header)

    def test_existing_per_view_results_backfill_novel_view_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir)
            results = {
                f"ours_{iteration}": {"PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.2}
                for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS
            }
            per_view = {}
            image_names = [f"{index:05d}.png" for index in reversed(range(18))]
            for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS:
                per_view[f"ours_{iteration}"] = {
                    "PSNR": {name: float(int(name[:5])) for name in image_names},
                    "SSIM": {name: float(int(name[:5])) / 100.0 for name in image_names},
                    "LPIPS": {
                        name: 1.0 - float(int(name[:5])) / 100.0 for name in image_names
                    },
                }
            (model_root / "results.json").write_text(json.dumps(results))
            (model_root / "per_view.json").write_text(json.dumps(per_view))

            self.assertTrue(
                run_omniscene.ensure_metric_scopes(
                    model_root, run_omniscene.CENTER150_EVAL_ITERATIONS
                )
            )
            updated = json.loads((model_root / "results.json").read_text())
            entry = updated["ours_1000"]
            self.assertEqual(entry[run_omniscene.ALL_VIEWS_SCOPE]["num_views"], 18)
            self.assertEqual(entry[run_omniscene.NOVEL_VIEWS_SCOPE]["num_views"], 12)
            self.assertAlmostEqual(entry[run_omniscene.NOVEL_VIEWS_SCOPE]["PSNR"], 5.5)
            self.assertAlmostEqual(entry[run_omniscene.NOVEL_VIEWS_SCOPE]["SSIM"], 0.055)
            self.assertAlmostEqual(entry[run_omniscene.NOVEL_VIEWS_SCOPE]["LPIPS"], 0.945)
            self.assertEqual(
                run_omniscene.completed_iterations(
                    model_root, run_omniscene.CENTER150_EVAL_ITERATIONS
                ),
                run_omniscene.CENTER150_EVAL_ITERATIONS,
            )
            self.assertFalse(
                run_omniscene.ensure_metric_scopes(
                    model_root, run_omniscene.CENTER150_EVAL_ITERATIONS
                )
            )

    def test_completed_legacy_experiment_only_updates_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model_root = output_dir / "001_scene"
            model_root.mkdir()
            results = {
                f"ours_{iteration}": {"PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.2}
                for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS
            }
            per_view = {
                f"ours_{iteration}": {
                    metric: {
                        f"{view:05d}.png": float(view)
                        for view in range(run_omniscene.EXPECTED_TEST_VIEWS)
                    }
                    for metric in run_omniscene.METRIC_NAMES
                }
                for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS
            }
            (model_root / "results.json").write_text(json.dumps(results))
            (model_root / "per_view.json").write_text(json.dumps(per_view))
            (model_root / "training_times.json").write_text(
                json.dumps(
                    {
                        "elapsed_seconds": {
                            str(iteration): float(iteration)
                            for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS
                        }
                    }
                )
            )
            args = argparse.Namespace(output_dir=output_dir, dry_run=False)
            loader = mock.Mock()

            with mock.patch.object(run_omniscene, "train_if_needed") as train, mock.patch.object(
                run_omniscene, "evaluate_missing_iterations"
            ) as evaluate:
                status = run_omniscene.process_scene(
                    0,
                    "token",
                    "001_scene",
                    loader,
                    run_omniscene.CENTER150_EVAL_ITERATIONS,
                    args,
                )

            self.assertEqual(status, "metrics-updated")
            train.assert_not_called()
            evaluate.assert_not_called()
            loader.assert_not_called()

    def test_new_evaluation_adds_both_metric_scopes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir)
            for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS:
                method_dir = model_root / "test" / f"ours_{iteration}"
                gt_dir = method_dir / "gt_1"
                render_dir = method_dir / "test_preds_1"
                gt_dir.mkdir(parents=True)
                render_dir.mkdir()
                for view in range(run_omniscene.EXPECTED_TEST_VIEWS):
                    name = f"{view:05d}.png"
                    (gt_dir / name).write_bytes(b"gt")
                    (render_dir / name).write_bytes(b"render")

            def write_metric_outputs(command, env, dry_run):
                self.assertEqual(command[1], "metrics.py")
                results = {
                    f"ours_{iteration}": {"PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.2}
                    for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS
                }
                per_view = {
                    f"ours_{iteration}": {
                        metric: {
                            f"{view:05d}.png": float(view)
                            for view in range(run_omniscene.EXPECTED_TEST_VIEWS)
                        }
                        for metric in run_omniscene.METRIC_NAMES
                    }
                    for iteration in run_omniscene.CENTER150_EVAL_ITERATIONS
                }
                (model_root / "results.json").write_text(json.dumps(results))
                (model_root / "per_view.json").write_text(json.dumps(per_view))

            args = argparse.Namespace(dry_run=False)
            with mock.patch.object(
                run_omniscene, "run_command", side_effect=write_metric_outputs
            ) as run_command:
                run_omniscene.evaluate_missing_iterations(
                    0,
                    model_root,
                    run_omniscene.CENTER150_EVAL_ITERATIONS,
                    args,
                )

            self.assertEqual(run_command.call_count, 1)
            self.assertEqual(
                run_omniscene.completed_iterations(
                    model_root, run_omniscene.CENTER150_EVAL_ITERATIONS
                ),
                run_omniscene.CENTER150_EVAL_ITERATIONS,
            )

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
                f"ours_{iteration}": self._metric_entry()
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

    def test_dispatch_does_not_sleep_between_skipped_scenes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self._write_completed_scene(output_dir / "001_scene")
            self._write_completed_scene(output_dir / "002_scene")
            args = argparse.Namespace(
                output_dir=output_dir,
                gpus="0",
                max_workers=1,
                dry_run=False,
            )

            with mock.patch.object(run_omniscene.time, "sleep") as sleep:
                failures = run_omniscene.dispatch_jobs(
                    [("token_1", "001_scene"), ("token_2", "002_scene")],
                    mock.Mock(),
                    run_omniscene.CENTER150_EVAL_ITERATIONS,
                    args,
                )

            self.assertEqual(failures, [])
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
