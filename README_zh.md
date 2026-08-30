# 麻将牌离线识别与清点（Offline Mahjong Tile Recognizer）

给桌面上一整副麻将牌拍一张照片，自动完成**精确计数、花色分布、缺牌明细与智能提示**——**纯端侧离线**，无云端、无网络权限，数据不出手机。

Android 应用（Kotlin + Jetpack Compose + TensorFlow Lite），附带完整的 YOLO 训练管线（Python + Ultralytics）。

> English README: [README.md](README.md)

## 功能

- **单张照片一次识别**：拍一张桌面散放的整副牌（108 张制：仅万/筒/条，无字牌，一张照片全部入镜）
- **背面牌也计数**：蓝/绿牌背同样检出并计入总数
- **逐种报告**：总数对比期望张数（默认 108，可改）、万/筒/条分布、缺牌/多出明细（每种应 4 张）、整门缺失提醒
- **智能提示**："缺的牌可能在朝下的牌中，请翻开核对"
- **部分清点**：期望张数设为小于 108 时，报告自动附口径说明
- **历史记录**：最近 50 次识别本地保存（缩略图 + 报告全文），可回看/删除
- **完全离线**：TFLite CPU（XNNPACK）推理；应用**没有网络权限**

## 识别流程

```
CameraX 拍照（4000×3000）或相册选图
  → EXIF 校正
  → 切 1280×1280 切片，重叠 25%
  → 每片跑两个 int8 TFLite 检测器：
       正面  YOLO11l @1280（27 种牌面）
       牌背  YOLO11m @1280（牌背，混合域训练）
  → 坐标映射回原图，按类别全局 NMS（IoU 0.30）
  → 规则引擎（计数/缺牌/提示）
  → 标注图 + 中文报告 → 存入历史
```

当前部署模型（int8，合计约 45 MB）：

| 模型 | 架构 | 输入 | 职责 |
|---|---|---|---|
| `mahjong_11l_1280_int8.tflite` | YOLO11l | 1280×1280 | 正面牌，28 类 |
| `mahjong_11m_back_1280_int8.tflite` | YOLO11m | 1280×1280 | 牌背检测 |

1920 下一代（YOLO11x 正面 + 更大切片）训练中，见 `docs/GPU_TRAINING_zh.md`。

## 构建

要求：Android SDK（minSdk 26 / targetSdk 36）、JDK 17。Android Studio 或纯命令行均可。

```sh
./build.sh assembleDebug          # 或 ./gradlew assembleDebug
./build.sh testDebugUnitTest      # JVM 单元测试
```

APK 输出在 `app/build/outputs/apk/debug/app-debug.apk`（约 87 MB，含双模型）。
应用界面为中文；把牌均匀铺开在桌面拍一张即可。

## 仓库结构

```
app/                  Android 应用（Compose 界面 / 相机 / ML 管线 / 规则引擎 / 历史记录）
training/             完整 YOLO 训练管线（Python + Ultralytics）
  synth/              合成场景生成器（随机旋转/透视/阴影/牌背混合）
  tools/              标注质检工具（标签对照表、几何规则修标）
  user/               个人照片入库脚本（模型辅助预标注）
docs/                 开发日志、训练记录、GPU 训练指南
```

## 训练管线

训练数据 = 公开数据集 + 大量合成数据（精灵裁片合成到桌面背景，随机旋转/透视/阴影/牌背）
+ 可选的个人实拍（**作者照片不随仓库发布**——把你自己的照片放进 `training/user/photos/`）。

亮点：

- `training/synth/generate.py` — 确定性合成场景生成器（`--canvas 640/1280/1920`）
- `training/user/prepare_face_dataset.py` — 自己照片的模型辅助预标注
  （切片 → 推理 → 跨切片按类 NMS → YOLO 标签 + 人工核验预览）
- `training/tools/label_sheet.py` — 标签对照表（contact sheet）核验工具
- `training/tools/fix_tong_circles.py` — 几何规则修标工具（Hough 圆 + 圆心去重，
  区分"2 大圆"与"4 小圆"筒子）
- `training/run_gpu_round.sh` — GPU 盒子上一键跑完整训练轮
  （预训 → 高分辨率微调 → int8 导出），见 `docs/GPU_TRAINING_zh.md`

部署的 1280 代验证指标：正面 mAP50 0.985 / 牌背 mAP50 0.988（完整记录见
`docs/TRAINING_RECORD_DGX.md`）。注意验证集 mAP ≠ 真实计数准确率；本项目的验收
方式是小型个人实拍回归集 + 人工核对。

## 隐私

- 仓库**不含任何个人照片**。作者照片只存在于作者本机（已 git-ignore）；发布的模型
  微调时用到过它们，但图片本身不随仓库分发。
- 应用只申请相机权限（相册选图走系统选择器），**没有 INTERNET 权限**。

## 致谢

训练数据基于 Roboflow Universe 的公开麻将检测数据集，以及用于合成拼贴的
[FluffyStuff 麻将牌面精灵](https://github.com/fluffyshrimp/mahjong-tiles)。感谢开源社区：
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)、TensorFlow Lite、Jetpack Compose。

## 许可

**双许可。**

- **开源使用：GNU AGPL-3.0**（见 [LICENSE](LICENSE)）。个人、学术及任何其他使用免费——
  但衍生作品（包括网络服务部署）必须以 AGPL-3.0 开源。
- **商业许可**：如需在闭源产品中使用本项目而不承担 AGPL 义务，请通过 Issue 联系作者
  购买商业许可。
