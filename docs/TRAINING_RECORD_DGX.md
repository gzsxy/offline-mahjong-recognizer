# DGX Spark 训练记录（B 阶段 M7，2026-08-30 完成）

机器：NVIDIA GB10（DGX Spark，aarch64，CUDA 13.0）· torch 2.13.0+cu130 · ultralytics 8.4.108

## 四个 run 最终结果

| run | 轮次 | 早停（patience） | 最佳轮次 | 最佳 mAP50 | 最佳 mAP50-95 | fitness 口径 |
|---|---|---|---|---|---|---|
| mj-face-11l-640 | 60（含 Mac 断点 10 轮，本机续训 50 轮，9.0h） | 未触发（15） | e55 | 0.9918 | 0.8599 | 0.2×mAP50+0.8×mAP50-95 |
| mj-back-11m-640 | 60 全新（1.5h） | 未触发（15） | e60 | 0.9925 | 0.9629 | 同上 |
| mj-face-11l-1280 | 12 跑满（3.8h） | 未触发（5） | e11 | 0.9847 | 0.8791 | 同上 |
| mj-back-11m-1280 | 12 跑满（0.6h） | 未触发（5） | e12 | 0.9884 | 0.9311 | 同上 |

- 起始权重：1280 两段分别来自对应 640 run 的 best.pt（e55 / e60）。
- 1280 的 mAP 与 640 不可直接横比：验证集换成 1280 原生混合集（face 含 mj-v2 30% 回放 val，back 为 back-synth-1280 + user-back-1280 val）。
- 1280 超参：imgsz 1280、batch 8、workers 8、lr0 0.0005、lrf 0.01、warmup 3.0、patience 5、close_mosaic 10、seed 44、deterministic、fliplr 0.0，其余增强照抄 640 脚本。

## 训练数据

- 640：face = mj-v2（15685/2903）；back = mj-back-mix-v1（3024/648，setup_portable_env.py 重建符号链接）。
- 1280：`mj-face-1280-mix-v1` train 7218 = mj-synth-1280-v1 2500 + mj-user-face-v1 12 + mj-v2 随机 30% 回放 4706（seed 44）；val 1273（回放比例 val 侧同为 30%）。`mj-back-1280-mix-v1` train 1524 = back-synth-1280 1500 + user-back-1280 24；val 258。构建脚本 `training/build_1280_mix.py`，零断链。

## 迁移/续训修复记录（Mac → DGX）

1. last.pt 内嵌 train_args 带 Mac 绝对路径（save_dir 等），resume 建目录即炸：`resume_face_11l_640.py` 显式覆盖 save_dir/data/device（ultralytics resume 用 checkpoint 内嵌参数，args.yaml 不起作用；仅 save_dir/device 等白名单键可覆盖，data 在内嵌路径不存在时回落）。
2. MPS 断点的 GradScaler 状态为空（MPS 上禁用），CUDA+AMP 启用态 scaler 拒绝空状态：`training/fix_resume_scaler.py` 将 ckpt["scaler"] 替换为本机新建启用态（scale 65536 起步），原文件备份 `mj-face-11l-640/weights/last.pt.mps-bak`（未随包携带，保留在本机）。
3. 断点实际为 10 个完整 epoch（迁移说明写 11），最后一轮在 Mac 上被中断未保存，不影响续训。
4. 解包产生的 2.5 万个 macOS `._*` AppleDouble 文件已清理（否则污染数据扫描与符号链接合并）。

## 本机产物

- run 目录：training/runs/{mj-face-11l-640, mj-back-11m-640, mj-face-11l-1280, mj-back-11m-1280}
- 日志：training/runs/b640_pretrain.log（Mac 旧日志存档 b640_pretrain.mac-20260829.log）、b1280_finetune.log
- 脚本：resume_face_11l_640.py（已改）、fix_resume_scaler.py、build_1280_mix.py、train_face_11l_1280.py、train_back_11m_1280.py、run_1280_finetune.sh

## Mac 侧后续（不在 DGX 做）

int8 TFLite 导出（imgsz=1280、cpu）→ ImageSlicer 切 1280 → 替换 assets 三模型 → 构建 APK → 实拍回归对比 → 回填 开发文档.md §10.5。导出用各 run 的 best.pt。
