# Offline Mahjong Tile Recognizer

**[中文说明](README_zh.md)**

Photograph a full set of mahjong tiles on a table — get an exact count, suit distribution, missing-tile report and smart hints, **entirely on-device**. No cloud, no network permission, no data leaves your phone.

An Android app (Kotlin + Jetpack Compose + TensorFlow Lite) with a complete YOLO training pipeline in Python (Ultralytics).

## Features

- **Single-photo recognition**: one shot of a spread-out 108-tile set (Chinese suited tiles only: 万/筒/条, no honors), fully covered in frame
- **Counting that includes face-down tiles**: blue/green tile backs are detected and counted too
- **Per-tile report**: count vs. expected (default 108, configurable), suit distribution, missing/excess tiles (each kind should have 4), whole-suit missing warnings
- **Smart hints**: "the missing tiles may be among the N face-down tiles — flip and check"
- **Partial-count mode**: set expected < 108, report adapts
- **History**: last 50 recognitions stored locally (thumbnail + full report), review/delete
- **Fully offline**: TFLite CPU (XNNPACK) inference; no INTERNET permission at all

## How it works

```
CameraX photo (4000×3000) or gallery pick
  → EXIF correction
  → slice into 1280×1280 tiles, 25% overlap
  → two int8 TFLite detectors per slice:
       face  YOLO11l @1280  (27 suited classes)
       back  YOLO11m @1280  (tile backs; mixed-domain trained)
  → map back to full image, per-class global NMS (IoU 0.30)
  → rule engine (count / shortage / hints)
  → annotated image + Chinese text report → saved to history
```

Current deployed models (int8, ~45 MB total):

| model | architecture | input | purpose |
|---|---|---|---|
| `mahjong_11l_1280_int8.tflite` | YOLO11l | 1280×1280 | face-up tiles, 28 classes |
| `mahjong_11m_back_1280_int8.tflite` | YOLO11m | 1280×1280 | tile backs |

A 1920-generation (YOLO11x face + larger slices) is in progress — see `docs/GPU_TRAINING_zh.md`.

## Build

Requirements: Android SDK (minSdk 26 / targetSdk 36), JDK 17. Android Studio or plain CLI both work.

```sh
./build.sh assembleDebug          # or: ./gradlew assembleDebug
./build.sh testDebugUnitTest      # JVM unit tests
```

APK lands at `app/build/outputs/apk/debug/app-debug.apk` (~87 MB, includes both models).
The app is in Chinese; point a camera at tiles spread out on a table and shoot.

## Repository layout

```
app/                  Android app (Compose UI / camera / ML pipeline / rule engine / history)
training/             Full YOLO training pipeline (Python + Ultralytics)
  synth/              Synthetic scene generator (random rotation/perspective/shadow/back mixing)
  tools/              Label QA tooling (contact sheets, geometry-based label fixes)
  user/               Personal-photo ingestion scripts (model-assisted pre-annotation)
docs/                 Development log, training records, GPU training guide (Chinese)
```

## Training pipeline

Training data = public datasets + a heavy synthetic generator (sprite cutouts composited onto
table backgrounds with random rotation/perspective/shadows/backs) + optional personal photos
(**author's photos are NOT part of this repository** — put your own into `training/user/photos/`).

Highlights:

- `training/synth/generate.py` — deterministic synthetic scene generator (`--canvas 640/1280/1920`)
- `training/user/prepare_face_dataset.py` — model-assisted pre-annotation of your own photos
  (slice → predict → cross-slice class NMS → YOLO labels + human-review previews)
- `training/tools/label_sheet.py` — contact-sheet tool for reviewing labels
- `training/tools/fix_tong_circles.py` — geometry-based label fixer (tells 2-big-circle vs
  4-small-circle tiles apart with Hough circles + center dedup)
- `training/run_gpu_round.sh` — one-shot orchestration for a full training round on a GPU box
  (pretrain → high-res finetune → int8 export), see `docs/GPU_TRAINING_zh.md`

Validation numbers of the deployed 1280 generation: face mAP50 0.985 / back mAP50 0.988
(full record in `docs/TRAINING_RECORD_DGX.md`). Note that validation mAP ≠ real-world counting
accuracy; the project's acceptance process is a small personal photo regression set + manual review.

## Privacy

- The repository contains **no personal photos**. Author's photos live only on the author's machine
  (git-ignored); the published models were fine-tuned with them, but the images themselves are not
  distributed.
- The app itself requests only the CAMERA permission (gallery picks use the system picker) and has
  **no INTERNET permission**.

## Acknowledgments

Training data builds on public mahjong detection datasets from Roboflow Universe and the
[FluffyStuff mahjong tile sprites](https://github.com/fluffyshrimp/mahjong-tiles) used for synthetic
composition. Thanks to the open-source community — [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
TensorFlow Lite, Jetpack Compose.

## License

**Dual licensed.**

- **Open source: GNU AGPL-3.0** (see [LICENSE](LICENSE)). Personal, academic and any other use is
  free — but derivatives (including network-service deployments) must be released under AGPL-3.0.
- **Commercial licensing**: if you want to use this project in a closed-source product without
  AGPL obligations, contact the author (open an issue) for a commercial license.
