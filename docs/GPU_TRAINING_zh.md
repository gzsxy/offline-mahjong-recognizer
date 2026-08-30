# GPU 盒子训练说明 — 1920 轮（C 阶段，2026-08-30）

分工：**训练与 TFLite 导出在 GPU 盒子**（优先 L40，其次 DGX Spark / 其他 CUDA 机器）；Mac 负责整合进 App + 桌面回归；用户负责真机验证。目标规格见 `开发文档.md` §4.1-C。

打包文件：`majiang-training-1920-20260830.tar`
SHA256：见文末

## 1. 包里有什么

- `training/datasets/`：mj-v2（640 预训集）、mj-user-face-1920-v1、mj-user-back-1920-v1（用户实拍 1920 切片，标签已经几何规则 + 人工终审修正）
- `training/runs/mj-back-11m-640/`：牌背 1920 微调的起始权重（60 epochs 训好的 11m）
- `training/runs/pretrained/yolo11x.pt`：COCO 预训练权重
- `training/`：全部脚本（含本轮新增 train_face_11x_640/1920、train_back_11m_1920、build_1920_assets、run_gpu_round、setup_portable_env）
- `training/synth/generate.py`：合成数据生成器（支持 --canvas 1920；**包内不含 1920 合成图**，编排脚本会自动生成，CPU 数小时）
- `.git` + `开发文档.md` + `TRAINING_RECORD_DGX.md`：完整上下文

## 2. 环境（aarch64 或 x86_64 Linux + NVIDIA CUDA）

```sh
tar -xf majiang-training-1920-20260830.tar -C ~/majiang && cd ~/majiang
python3 -m venv training/venv
training/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
training/venv/bin/pip install ultralytics==8.4.108 opencv-python pyyaml
```

- **ultralytics 锁 8.4.108**（与现有 best.pt checkpoint 同版本）。
- pip 装 CUDA torch 失败就用 NGC 官方 PyTorch 容器。
- 解包目录确定后不要再移动（data.yaml 会写绝对路径）。

## 3. 一键开跑

```sh
nohup sh training/run_gpu_round.sh > training/runs/gpu_round.log 2>&1 &
tail -f training/runs/gpu_round.log
```

编排脚本自动依次执行：本机化 → 生成 1920 合成数据（缺才生成）→ 构建混合集/校准集 → **face YOLO11x@640 预训 60ep** → **face 11x@1920 微调 12ep** → **back 11m@1920 微调 12ep** → 导出两枚 int8 TFLite（imgsz=1920）。

## 4. 注意事项

1. **face 11x@1920 是显存大头**：batch 4 约需 40–60GB；OOM 就把 `training/train_face_11x_1920.py` 的 batch 降到 2（epoch 数不变）。L40（48GB）按 batch 4 起步。
2. 断点续训：预训中断后把 `train_face_11x_640.py` 换成 `YOLO('training/runs/mj-face-11x-640/weights/last.pt').train(resume=True)` 方式（参照 training/resume_face_11l_640.py 的 save_dir/device 覆盖做法）。
3. `fliplr=0.0`、seed、deterministic 不要动（牌面有方向性）。
4. 1920 用户切片来自 1704×1279 的小图，经过了非等比拉伸——比例很小（14/7200+），不影响大局，勿再放大该数据权重。
5. 早停：mAP 连续 5 轮（微调）/15 轮（预训）无提升可停，记录停在哪轮。
6. 完成后**不要删任何 run 目录**。

## 5. 完成后传回 Mac

```
training/runs/mj-face-11x-640/    （weights/ + results.csv + args.yaml）
training/runs/mj-face-11x-1920/   （同上）
training/runs/mj-back-11m-1920/   （同上）
两枚导出的 *int8.tflite（若 step6 成功）
```

传回后交给 Mac 侧 ZCode：整合（assets + ImageSlicer 1920）→ 桌面回归 → APK → 用户真机验证 → 文档回填。

## 6. 给 GPU 盒子上 ZCode 的接手指令（可直接粘贴）

> 项目在当前目录（麻将牌识别 1920 轮训练包）。先读 `开发文档.md` §4.1-C 与 `GPU训练说明.md`。职责只有训练与导出：
> 1. 按说明第 2 节装环境（ultralytics 锁 8.4.108）。
> 2. `nohup sh training/run_gpu_round.sh > training/runs/gpu_round.log 2>&1 &` 后台执行全流程；监控 log，face 11x@1920 若 OOM 将 batch 降 2 重跑该段。
> 3. 各阶段 mAP 记录到 training/runs/gpu_round.log 即可；微调段 mAP 连续 5 轮无提升可提前停（记下停在哪轮）。
> 4. 完成后按说明第 5 节列出的文件清单打包，等待用户传回 Mac。不要改 App 侧文件、不要做回归。

---

校验（Mac 上生成）：
- 文件：`majiang-training-1920-20260830.tar`
- SHA256：见打包输出
