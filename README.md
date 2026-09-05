# BVLab-Analyzer User Guide

A PyQt5-based tool for analyzing 3D biomedical images (TIF / BV / OME-Zarr), supporting annotation, model training, prediction, and morphological statistics for cell, neural, and vessel structures.

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / 11 | Windows 11 |
| CPU | 6 cores | 16+ cores |
| GPU | NVIDIA, CUDA 6.1+, VRAM ≥8GB | CUDA 8.9+, VRAM ≥24GB |
| RAM | 64GB | 128GB+ |
| Python | 3.10 | 3.10 |

Install dependencies:

```bash
pip install -r requirements.txt
```

## Launch

```bash
python BVLabAnalyzer.py
```

## Documentation and Tutorial

Alongside this quick start guide, a comprehensive user manual is available in `BVLab_Analyzer_user_guide_en.docx`. For a visual walkthrough, a screen recording video illustrating the typical analysis pipeline is provided as `Video of the typical analysis pipeline.mp4`. We recommend reviewing both resources before starting your first project.

## Data Formats

- Images: 3D TIF (recommended), BV format (folder must contain `config.cfg`), OME-Zarr (`*.ome.zarr`).
- Labels: SWC point cloud files (standard SWC specification, see appendix).

## Three Annotation Targets

| Type | Annotation target | Output |
|---|---|---|
| Cell | Center point of each cell | Single-point SWC (no segments) |
| Neural | Centerline of neurite skeleton | Tree-like SWC |
| Vessel | Centerline of vessel network | Network SWC (supports branching) |

## Core Workflows

- **Path A** (train a new model, recommended): Dataset creation → Model training → Image prediction → Result revision → Statistics.
- **Path B** (with existing labels): Dataset creation → Model training → Image prediction → Statistics.
- **Path C** (pretrained model only): Image prediction → Result revision.
- **Path D** (manual annotation only): Enter the revision module and annotate manually.

## Module Overview

### 1. Dataset Creation
Converts raw images and SWC labels into training format (`config.json` + data patches). Supports parameters such as filtered block count, parallel process count, training patch size, train/validation/test ratio, and BV ROI. Outputs to `MakeResults/`.

### 2. Model Training
Select `config.json` and a save path, set training epochs (default 500, with early stopping). Supports resuming from checkpoints and real-time loss/evaluation curves. Outputs `cell_train.pth / neural_train.pth / vessel_train.pth`.

### 3. Image Prediction
Select a config/model, image path, and save path to auto-annotate TIF/BV/OME-Zarr data. For large-scale data, use **Result Splicing** to merge the SWC results.

### 4. Result Revision (Manual Annotation)
Two modes: **Browse mode** (default) and **Revision mode** (press `V` to enter).

- Cell: left-click to add a point / right-click to delete.
- Neural / Vessel: left-click two points to interpolate a line / right-click for a single point / Alt+click to connect branches.

Common shortcuts:

| Action | Cell | Neural / Vessel |
|---|---|---|
| Save | Ctrl+S | Shift+S |
| Open file | Ctrl+Shift+Q | Ctrl+Shift+Q |
| Clear all annotations | C | C |
| Undo | Ctrl+Z | Ctrl+Z |
| Enter/exit revision | V (toggle) | V enter / S exit |
| Connect branches | - | Alt+left-click two endpoints |

### 5. Statistics (Neural / Vessel)
Input images, SWC labels, and a config file; set resolution and units. Outputs per-segment statistics (CSV) and network-wide statistics (XLSX), with charts for volume, length, tortuosity, and radius.

## Main Directories

| Directory | Description |
|---|---|
| `BVLabAnalyzer.py` | Program entry point |
| `DataMake/` | Dataset creation (BV / Zarr patch sampling) |
| `DataTrain/` | Model training |
| `DataPredict/` | Model inference |
| `DataSetMake/` | Dataset generation |
| `DataStatistics/` | Result statistics |
| `EasyTracing/` | Skeletonization, SWC generation and smoothing |
| `ImageCut/`, `ImageSplice/` | Large-image cropping and stitching |
| `SwcToMask/` | SWC to mask conversion |
| `ViewWidget/`, `cell_points_marking/`, `line_points_marking/`, `vessel_lines_marking/` | Visualization and annotation |
| `BVExample/` | BV data reading |

## Appendix: SWC Format

One point per line:

```
index type X Y Z radius parent
```

- Type: 0=undefined, 1=soma, 2=axon, 3=dendrite, 4=branch point.
- Coordinates are in voxel units; parent index `-1` indicates a root node.

Example:

```
1 1 10.5 20.0 5.0 2.5 -1
2 3 12.0 21.0 6.0 2.0 1
3 3 14.0 22.0 7.0 2.0 2
```

## Troubleshooting

- `CUDA out of memory` during prediction: reduce the training patch size or use a GPU with more VRAM.
- Shortcuts not working in revision mode: click the image area first to give the visualization window focus.
- Incorrect statistics: check whether the resolution parameter matches the physical size of the image.
