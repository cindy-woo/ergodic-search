# Ergodic Search with Adaptive Replanning

This repository contains experiments for **ergodic search in dynamic information maps** using a receding-horizon trajectory optimizer in PyTorch.

The main script (`test.py`) plans trajectories over entropy/information maps, detects map changes, and selectively replans to reduce compute while keeping strong coverage behavior.

## Slides

- Google Slides: https://docs.google.com/presentation/d/1uBERaze0npAR-7jfm4VOMRwVCuqiUuYrLSDtvhGjmDo/edit?usp=sharing
- Local deck snapshot: `Meeting of Minds - Presentation.pdf`

## Repository Highlights

- `test.py`: Main adaptive ergodic planning + visualization pipeline
- `baseline.py`: Baseline method for comparison
- `compare_methods_test.py`: Comparison/evaluation script
- `tuning/`: Hyperparameter sweep/tuning utilities
- `entropy_maps/`: Input map set used in experiments
- `comparison_outputs/`: Generated comparison plots and CSV metrics
- `simulation_screenshots/`: In-simulation captures (step-by-step views, top-down snapshots, and run diagnostics)
- `updated_trajectories/`: Curated trajectory figures from newer planner versions/settings

## Method Summary

`test.py` combines:

1. Fourier-based ergodic objective (trajectory statistics vs. map distribution)
2. Hotspot extraction from high-information regions
3. Ordered-goal guidance (`h1 -> h2 -> h3`) for structured coverage
4. Sensor activation optimization (`lambda`) coupled to local information
5. Adaptive replanning policy based on map-change tiers:
   - unchanged
   - small
   - medium
   - large

Instead of fully replanning every cycle, the script warm-starts or skips heavy optimization when possible.

## Requirements

Python 3.10+ recommended.

Install core dependencies:

```bash
pip install torch numpy scipy matplotlib
```

## Quick Start

Run the main adaptive planner:

```bash
python test.py
```

What it does:

- Loads maps from `entropy_maps/`
- Builds a multi-cycle map sequence (with optional repeats)
- Runs adaptive replanning over cycles
- Prints cycle diagnostics (replan reason, optimizer iterations, timing)
- Displays trajectory and hotspot visualizations

## Key Parameters (in `test.py`)

- Planning horizon: `T_HORIZON`
- Executed head length per cycle: `HEAD_STEPS`
- Optimization budget: `ITERS_FULL`, `ITERS_WARM`, `ITERS_REFINE`
- Replanning thresholds: `UNCHANGED_EPS`, `SMALL_CHANGE_THR`, `LARGE_CHANGE_THR`
- Sensor behavior: `SENSOR_ON_THRESHOLD`, `SENSOR_*` weights
- Hotspot behavior: `MAX_HOTSPOTS`, `HOTSPOT_QUANTILE`, `MIN_HOTSPOT_AREA`

## Typical Outputs

- On-screen trajectory figures (head/tail split + sensor-lambda coloring)
- Hotspot extraction visualizations
- CSV/plot artifacts under comparison folders for method analysis

## Notes

- Current paths in scripts assume local repo location; if needed, update `entropy_maps_path` in `test.py`.
- Randomness is enabled by default (`RUN_SEED = None`); set a fixed seed for reproducibility.
