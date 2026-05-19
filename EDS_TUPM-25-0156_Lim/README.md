# EDS_TUPM-25-0476_Legaspi
## ROB-03: Kinematic Repeatability Error — Engineering Data Systems Pipeline
**Course:** Computer Programming | **Academic Year:** 2026

---

## Project Overview
This project implements a fully automated Python data analytics pipeline to analyze **kinematic repeatability error** in a 6-DOF industrial robot arm. It applies statistical analysis and engineering-grade visualization on a filtered subset of a robot inverse kinematics dataset.

**Assigned Topic:** ROB-03 — Kinematic Repeatability Error
**Engineering Pillar:** Pillar 6 — Robotics & Mechatronics

---

## Unique Filter Logic
The dataset is filtered to isolate the **Elevated-Elbow, High-Reach Wrist Configuration**:
- `q3 > 1.572` — Elbow joint above median angle
- `z > 0.30` — End-effector in elevated Z zone (above 0.30 m)
- `3.0 ≤ q6 ≤ 5.5` — Wrist rotation in mid-to-high sector

This retains **1,395 rows** (9.30%) from the original 15,000-row dataset.

---

## Repository Structure
```
EDS_TUPM-25-0476_Legaspi/
├── main.py                   # Full Python pipeline (OOP, 4 classes)
├── requirements.txt          # Required libraries
├── data/
│   ├── dataset_original.csv  # Raw Kaggle dataset
│   └── dataset_cleaned.csv   # Filtered and cleaned dataset
├── outputs/
│   ├── static_1_histogram.png
│   ├── static_2_boxplot.png
│   ├── static_3_scatter.png
│   ├── static_4_heatmap.png
│   ├── animation_1_rolling_error.gif
│   └── animation_2_3d_path.html
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AlbertWayne/EDS_TUPM-25-0476_Legaspi.git
cd EDS_[StudentNum]_[Surname] # EDS_TUPM-25-0476_Legaspi (File Name)
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place the dataset
Copy `robot_inverse_kinematics_dataset.csv` into the `data/` folder and rename it `dataset_original.csv`.

### 4. Run the pipeline
```bash
python main.py
```

All outputs will be saved automatically to the `data/` and `outputs/` folders.

---

## Pipeline Architecture
| Class | Responsibility |
|---|---|
| `DataIngestion` | Load CSV, apply unique filter, error handling |
| `DataCleaner` | Remove nulls/duplicates, fix types, feature engineering |
| `Analyzer` | NumPy/SciPy descriptive, distribution, correlation, comparative stats |
| `Visualizer` | 4 static charts + 2 animations (Matplotlib + Plotly) |

---

## Dataset
- **Source:** Robot Inverse Kinematics Dataset (Kaggle)
- **Columns:** `q1–q6` (joint angles, rad), `x`, `y`, `z` (end-effector position, m)
- **Engineered Feature:** `position_error` — Euclidean distance of end-effector from origin (m)
