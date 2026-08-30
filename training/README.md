# Training

The checked-in Android models are generated from:

`runs/mj-synth-v1/weights/best_int8.tflite`

The original dataset directories referenced by the training run metadata are
outside this repository, so the training dataset itself is not reproducible
from the checked-in files yet. The run artifacts and model are kept here for
Android-side validation and later retraining.

## Validate A Model

Run this from the project root after installing the training environment:

```sh
training/venv/bin/python training/validate_model.py
```

To inspect a particular model:

```sh
training/venv/bin/python training/validate_model.py \
  --model training/runs/mj-synth-v1/weights/best_int8.tflite
```

## User-Specific Fine-Tuning

The current user photos contain four 12x10 blue/green back grids. The
preparation script creates labels from the checked grid corners, crops them at
the same scale used by Android, and adds photometric augmentation:

```sh
training/venv/bin/python training/user/prepare_back_dataset.py \
  --source "$HOME/Downloads"
training/venv/bin/python training/user/prepare_finetune_dataset.py
training/venv/bin/python training/user/finetune.py
```

For scattered blue/green backs, the shorter `finetune_back.py` pass adapts the
verified v1 back detector with real user textures without changing the public
face detector:

```sh
training/venv/bin/python training/user/finetune_back.py
```

The fine-tuning model is used only for `back` detections. The public-data
`mj-v2` model remains responsible for face-up tiles, preventing catastrophic
forgetting when the real photos contain only backs. Export both models before
copying them to `app/src/main/assets/`:

```sh
training/venv/bin/python -c "from ultralytics import YOLO; YOLO('training/runs/mj-v2/weights/best.pt').export(format='tflite', imgsz=640, int8=True, data='training/datasets/mj-v2/data.yaml', nms=False, device='cpu')"
training/venv/bin/python -c "from ultralytics import YOLO; YOLO('training/runs/mj-user-v3/weights/best.pt').export(format='tflite', imgsz=640, int8=True, data='training/user/back-only.yaml', nms=False, device='cpu')"
```

For the complete face model, first build the 43-class merged dataset and
expand the existing face checkpoint. This keeps the learned suited-tile and
back head weights while adding honors and flower/season classes:

```sh
training/venv/bin/python training/tools/remap_merge.py --full \
  --out training/datasets/mj-full-v1
training/venv/bin/python training/tools/expand_checkpoint.py
training/venv/bin/python training/train_full.py
```

The training environment uses the matched pair `torch==2.12.1` and
`torchvision==0.27.1`. An incompatible torchvision build can fail before the
first epoch with `torchvision::nms does not exist`.
