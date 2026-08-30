mahjong_11l_1280_int8.tflite

- 用途：正面牌检测（B 阶段新一代主模型）
- 来源：training/runs/mj-face-11l-1280/weights/best_int8.tflite
  （YOLO11l，640 预训 60 epochs + 1280 微调 12 epochs，训练记录见 TRAINING_RECORD_DGX.md）
- 输入：NCHW [1, 3, 1280, 1280]，float32（int8 内部量化，权重动态量化）
- 输出：[1, 32, 33600]，float32；前 4 通道 0..1 归一化 cx/cy/w/h，后 28 通道类别分数
- 类别：27 种牌面 + back，共 28 类
- 体积约 25 MB；搭配 ImageSlicer(tileSize=1280, overlap 25%) 使用

mahjong_11m_back_1280_int8.tflite

- 用途：牌背检测（整齐网格 + 散放，混合域训练，替代旧双牌背模型与散放辅助模型）
- 来源：training/runs/mj-back-11m-1280/weights/best_int8.tflite
  （YOLO11m，640 预训 60 epochs + 1280 微调 12 epochs）
- 输入输出契约与上面相同；Ensemble 只采用该模型的 back 类结果
  （它在合成散放场景中同时学习过正面类，正面照片上的正面类输出会被按类过滤丢弃）
- 体积约 20 MB

部署口径：ImageSlicer 1280/25% 重叠，切片内 NMS 0.45，全局按类别 NMS 0.30，
正面 conf 0.45，牌背 conf 0.45，CPU/XNNPACK，双模型合计约 45 MB。
桌面回归（作者 11 张实拍，结果未随仓库发布）：牌背网格 125–127
（物理网格 13×10≈130，右缘数张被裁切），正面照片上新牌背模型跨误检 0–3（旧版 11–27）。
