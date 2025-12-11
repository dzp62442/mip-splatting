# OmniScene 数据集实验规划（mip-splatting）

> 目标：基于 depthsplat（前馈高斯）已有的 OmniScene 数据制作流程，在 mip-splatting（逐场景优化）中以 **Blender JSON** 的形式复用同样的 10 个 val bin，完成训练/渲染/评估自动化。所有中间结果写入项目输出目录，保证 `PATH_PROJECT/datasets/OmniScene` 原始 500GB 数据保持只读。

---

## 1. 背景：depthsplat 中的实现回顾
- **配置入口**：depthsplat 通过 Hydra 配置（`config/dataset/omniscene.yaml` + `experiment=omniscene_*`）注册 `DatasetOmniScene`，加载 `datasets/omniscene`（nuScenes 派生）。数据集一次性提供 *context*（6 张输入图 + 相机）和 *target*（包含输入在内共 18 张输出视图）。
- **数据划分与抽样**：train/val/test/demo 四个 stage 对应不同的 `bins_*.json`，val 默认抽取 10 个 bin（`[:30000:3000][:10]`）以便快速可视化。
- **数据字段**：`dataset_omniscene.py` 负责 resize（112x200/224x400）、更新 intrinsics、加载 mask，返回 `context`/`target` 字典，供前馈网络直接使用。
- **运行方式**：`python -m src.main` 由 Hydra 控制训练/验证/测试节奏，模型单次 forward 即完成场景预测（无 per-scene 优化循环）。

## 2. 与 mip-splatting 的差异
| 方面 | depthsplat（前馈） | mip-splatting（逐场景优化） |
| --- | --- | --- |
| 数据粒度 | batch 内含多个 bin，一次 forward 结束 | 每个场景需迭代 ~30k 次优化，逐场景处理 |
| Loader 输出 | 返回 tensors（context/target） | 需要磁盘结构（train/test 图像 + JSON）供 `Scene` 读取 |
| 调用入口 | Hydra + DataModule | 直接运行 `train.py`/`render.py`/`metrics.py` |
| 渲染/评估 | 模型内部 forward 完成 | 需显式调用渲染 & 评估脚本 |
| 场景划分 | loader 控制 batch | 需手动将 val 的 10 个 bin 映射为 10 个“伪场景” |

因此必须新增“桥接层”，将 depthsplat 的 data pipeline 转成 mip-splatting 可用的 Blender 风格数据。

## 3. 目录规划与主要组件
1. **`comp_svfgs/`（新建）**：存放 OmniScene 适配代码。
   - `omniscene_dataset.py`：读取原始数据、生成 Blender JSON、复制图像、缓存结果。
   - 后续若有辅助脚本（mask 处理、姿态转换等）也放在此目录。
2. **`docs/OmniScene数据集实验文档.md`**：持续更新设计与实现细节。
3. **`scripts/run_omniscene.py`**：仿照 `run_mipnerf360.py`，串联 loader、训练、渲染、评估。
4. **缓存/输出目录**：**代码** 与 **数据产出** 分开管理：
   - `comp_svfgs/` 仅放置适配脚本（如 `omniscene_dataset.py`），不再存储任何生成数据。
   - 所有中间转换结果与训练输出统一放在仓库根目录的 `output/` 下，便于清理与复用。默认结构：
   ```
   output/
     ├── omniscene_cache/
     │     ├── cache_112x200/
     │     │     └── scene_token/
     │     └── cache_224x400/
     └── omniscene_runs/             # 训练 + 渲染 + 评估结果
   ```
   - `cache_*` 保存转换后的 Blender 数据，既可作为 `train.py -s` 的输入，也可长期保留复用。
   - `omniscene_runs/<scene_token>/...` 用于记录 `train.py`/`render.py`/`metrics.py` 的所有产出。

## 4. 数据加载与转换细节（`comp_svfgs/omniscene_dataset.py`）
### 4.1 数据源
- 数据根：当前项目的 `datasets/OmniScene` 目录（已经通过符号链接/超链接指向真正的 500GB 原始数据）。后续脚本默认使用 `PATH_PROJECT/datasets/OmniScene`，若需切换可通过参数覆盖。
- 关键文件沿 depthsplat：
  - `interp_12Hz_trainval/bins_val_3.2m.json`
  - `interp_12Hz_trainval/bin_infos_3.2m/<token>.pkl`
  - `samples_small` 与 `sweeps_small`（RGB），`samples_mask_small` 与 `sweeps_mask_small`（动态掩码）。
- `OmniSceneLoader` 通过参数 `data_root` 指向此目录。

### 4.2 模式与抽样
- 默认 stage = `val`：`tokens = bins[:30000:3000][:10]`，10 个 bin 视为 10 个场景。
- 兼容 `train`（全量 `bins_train`）、`test`（mini-test 抽样）、`demo`（特定列表），方便后续扩展。

### 4.3 Blender JSON 结构
选择 **方案 B：落地 Blender transforms JSON**，以复用现有 `Scene` loader。

#### 4.3.1 文件布局
- `images_train/`: 6 张 key-frame 输入图像。
- `images_test/`: 18 张评估图像（6 张输入 + 12 张 novel views）。
- `transforms_train.json`, `transforms_test.json`: 对应 split 的相机列表。
- `meta.json`: 记录源 bin token、stage、生成时间、使用的分辨率等，方便排错。

#### 4.3.2 JSON 字段
```jsonc
{
  "camera_angle_x": <float>,          // 根据 resize 后的 fx 计算
  "frames": [
    {
      "file_path": "images_train/000000",
      "transform_matrix": [[...]],    // 4x4 C2W（OpenGL 坐标）
      "fl_x": <float>,                // 可选
      "fl_y": <float>,
      "cx": <float>,
      "cy": <float>,
      "w": 200,
      "h": 112,
      "near": 0.01,                   // 可选，统一写默认值
      "far": 100.0
    }, ...
  ]
}
```
实现要点：
- `transform_matrix` 必须符合 NeRF/Blender 约定：Y 轴向上、Z 轴向前。depthsplat 的 `load_info` 返回的 `c2w` 需要 `c2w[:3, 1:3] *= -1` 才能与 Blender loader 对齐。
- `camera_angle_x = 2 * atan(0.5 * width / fx)`，其中 `fx` 是 resize 后的值。
- `file_path` 不带扩展名，mip-splatting loader 会自动加 `.png`。
- 虽然 `Scene` 当前未使用 `near/far`，但写入固定值可以保持一致性。

### 4.4 图像与掩码处理
- 复用 depthsplat `load_conditions` 逻辑：读取 JPG、同步更新 intrinsics、输出 numpy/tensor。
- resize 尺寸：默认 112x200，可通过 `resolution_hw` 参数切到 224x400。
- 输出图像保存为 `PNG`（避免重复压缩）。命名规则：`{split}/{idx:06d}.png`。
- 动态掩码：当前 mip-splatting 未用到，可先不落地；若后续需要，可在 JSON 中增加 `mask_path` 字段或单独保存 `masks_train/`、`masks_test/`。

### 4.5 缓存策略
- `OmniSceneLoader.prepare_scene(token)`：
  1. 计算目标目录 `cache_root/cache_<reso>/<token>`。
  2. 若 `meta.json` 存在且与当前参数（stage、分辨率）匹配，则直接返回目录。
  3. 否则：
     - 读取 bin 信息、加载图像、生成 JSON；
     - 写入磁盘（可先写到临时目录再 `rename` 保证原子性）；
     - 写 `meta.json`，包含 `{"token": ..., "stage": ..., "resolution": [H,W], "train_frames": 6, "test_frames": 18}`。

## 5. 运行脚本 `scripts/run_omniscene.py`
1. **参数**（初步设定）：
   - `--data_root`（默认 `~/Projects/datasets/OmniScene/interp_12Hz_trainval`）
   - `--cache_root`（默认 `comp_svfgs/output_omniscene`）
   - `--resolution`（枚举 `112x200` / `224x400`）
   - `--stage`（默认 `val`）
   - `--output_dir`（训练结果保存目录，例如 `benchmark_omniscene`）
   - `--gpus`（可指定可用 GPU 列表，如 `0,1`）
   - `--max_workers`（线程池大小）
   - `--dry_run`
2. **流程**：
   1. 初始化 `OmniSceneLoader`，获取 token 列表（默认 10 个）。
   2. 遍历每个 token：
       - `scene_root = loader.prepare_scene(token)`（若缓存不存在则即时生成；若已有则直接复用）     
       - `model_root = os.path.join(output_dir, token)`
   3. 构造命令：
      ```
      train_cmd = (
        f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
        f"python train.py -s {scene_root} -m {model_root} --eval --white_background "
        f"--port {6009+gpu} --kernel_size 0.1 -r 1"
      )
      render_cmd = f"... python render.py -m {model_root} --skip_train"
      metrics_cmd = f"... python metrics.py -m {model_root} -r 1"
      ```
   4. 使用 GPUtil 获取空闲 GPU，将任务通过 `ThreadPoolExecutor` 提交，与 `run_mipnerf360.py` 相同的调度逻辑。
3. **运行方式**：
   - 单阶段完成。`run_omniscene.py` 会在每个场景开始前调用 `prepare_scene`：若缓存不存在则自动生成，若已存在则直接复用。因此只需运行一个脚本即可完成“数据转换 + 训练 + 渲染 + 评估”全流程。
   - 可通过参数（如 `--rebuild_cache`）显式要求重新生成缓存；默认为按需生成。

4. **分辨率控制**：
   - 因为我们提前 resize 到目标分辨率，`train.py` 里的 `-r` 固定设为 1 即可。
5. **结果产物**：
   - `model_root/point_cloud/iteration_30000/...`（高斯模型）
   - `model_root/test/ours_30000/...`（渲染结果）
   - `model_root/results.json`, `per_view.json`（metrics 输出）
   - 这些结构与现有脚本保持一致，便于对比分析。

## 6. 数据加载 vs. 逐场景流程的差异点（补充）
| 模块 | depthsplat 行为 | mip-splatting 需求 | 当前方案 |
| --- | --- | --- | --- |
| near/far | loader 决定，模型使用 | 可选 | JSON 中写入固定 0.01/100 以备未来使用 |
| mask | train.step 中用于剔除动态区域 | 暂无接口 | 初期忽略；若需要可扩展 render/metrics 支持 mask |
| 初始点云 | 模型内生成 | mip-splatting 读取 PLY；若无则随机 | Blender loader 若搜不到 `points3d.ply` 会自动生成随机点云，足够当前实验 |
| 分辨率 | loader 控制 | loader + run 脚本控制 | 通过 `--resolution` 参数设置，默认 112x200 |
| 存储 | 无需落地 | 必须写磁盘 | 使用 `cache_<reso>/<token>/` 结构 |

## 7. 实施步骤（最终版）
1. **实现 `comp_svfgs/omniscene_dataset.py`**：
   - 复制 depthsplat 中 `load_info`/`load_conditions` 的核心逻辑（仅保留图像+pose 部分）。
   - 实现 `OmniSceneLoader`，提供：`list_tokens()`, `prepare_scene(token)`。
   - 支持 112x200/224x400 两种分辨率、四种 stage。
2. **实现 `scripts/run_omniscene.py`**：
   - 解析参数，初始化 loader。
   - 顺序或并行调度各场景的 train/render/metrics。
   - 日志格式参考 `run_mipnerf360.py`。
3. **验证流程**：
   - 先在 `dry_run` 模式下检查命令/目录是否正确。
   - 选择一个 token，手动运行 train/render/metrics，确认工作流畅。
   - 批量运行 10 个场景，生成对比结果。
4. **文档更新**：
   - 实现后补充具体命令示例、生成目录截图、剩余 TODO（例如 mask、near/far 可调、迭代次数调参）。

---
本版本文档已细化数据落地格式、缓存策略与脚本流程，可直接指导后续编码实现。待用户审阅确认后即可着手开发。 
