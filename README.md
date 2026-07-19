# Multi-Modal AI System for Industrial Quality Assurance

An AI-powered inspection tool for steel surface defect detection. Upload an
image, get defects detected with bounding boxes via a trained YOLOv8 model,
and receive a professional inspection report generated locally by Llama 3.2
(via Ollama).

## Features

- **Defect detection** — YOLOv8m trained on the NEU-DET dataset (6 classes:
  crazing, inclusion, patches, pitted surface, rolled-in scale, scratches)
- **Adaptive preprocessing** — automatically detects whether an uploaded
  image is already in NEU-DET format (small, near-grayscale crops) or a
  general photo, and routes accordingly:
  - **NEU-format images** → direct single-pass inference
  - **General photos** → split into overlapping tiles, each individually
    contrast-normalized and sharpened, run through the model, then merged
    back via weighted box fusion
- **Local LLM reporting** — Llama 3.2 (via Ollama) turns raw detections into
  a structured inspection report: summary, severity, recommended actions
- **Streamlit web interface** — upload, detect, and review results in one
  page

# Setup

### 1. Install Python dependencies
```bash
pip install streamlit ultralytics pillow ollama
```

### 2. Install and configure Ollama
Download from [ollama.com/download](https://ollama.com/download), then:
```bash
ollama pull llama3.2
```
Ollama runs as a background service; ensure it's running before starting the
app (`ollama serve` if it hasn't auto-started).

### 3. Place the trained model
Copy `best.pt` (the trained YOLOv8m weights) into the project root, next to
`app.py`.

### 4. Run
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Model configuration

| Setting | Value | Reasoning |
|---|---|---|
| Base model | YOLOv8m (medium) | Best-performing variant of nano/small/medium/large tested; medium and large were compared in an ensemble, but the plain single-model medium checkpoint outperformed both ensembling and test-time augmentation in practice (see "Experiments" below) |
| Confidence — NEU-format images | 0.15 | Validated against real training-distribution images; genuine defects in this model's output commonly score 15–40%, so the YOLO default of 0.25 would discard true positives |
| Confidence — tiled/general photos | 0.35 | A low threshold combined with an already out-of-domain image produces near-uniform false positives across plain, defect-free regions; a stricter bar is used here to keep precision usable |
| Tile size / overlap | 200×200, 30% | Matches NEU-DET's native crop size; overlap ensures defects near a tile boundary are still fully captured in at least one tile |
| Box merging | Weighted box fusion | Overlapping detections from neighboring tiles are confidence-averaged into a single fused box, rather than picking one and discarding the rest (plain NMS) or leaving duplicates |

## Training performance

Trained for 40 epochs on NEU-DET. Final validation metrics:
- mAP50: 0.79
- mAP50-95: 0.49
- Precision: ~0.69
- Recall: ~0.75

The confusion matrix shows most misclassifications land in "background"
(missed detections) rather than confusing one defect class for another —
i.e. the model's main weakness is under-detection (recall), not
misclassification.

## Experiments tried (and why they were rejected)

In the interest of transparency, several inference-time accuracy
improvements were tested and *did not* improve results, so were not kept:

- **Test-time augmentation (TTA)** — reduced detection confidence on
  validated test cases rather than improving it
- **Multi-checkpoint ensembling** (nano/small/medium/large combined) —
  weaker checkpoints diluted confidence from the stronger medium model;
  even medium+large alone underperformed the single medium model
- **Looser NMS / higher tile overlap** — increased duplicate/fragmented
  detections without adding real signal

The final configuration (single YOLOv8m model, split confidence thresholds,
weighted box fusion) was the empirically best-performing setup found.

## Known limitation

The model performs reliably on images matching the NEU-DET training
distribution (validated via test images from the original dataset — see
demo video). Performance is visibly weaker on general photographs that
differ in lighting, scale, or capture angle — in particular, the
**scratches** class was not reliably detected on out-of-domain test photos,
despite reasonable validation-set performance (~75% correct on scratches
in the confusion matrix). This is a domain generalization gap common to
models trained on curated/lab-condition datasets, rather than a defect in
the detection pipeline itself. Addressing it fully would require additional
training data covering more varied real-world capture conditions.

## Submission contents

- `app.py` — Streamlit application
- `best.pt` — trained YOLOv8m weights
- `README.md` — this file
- Demo video — 2–5 minute walkthrough of the working application
