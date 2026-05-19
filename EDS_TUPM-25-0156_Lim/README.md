# EDS_TUPM-25-0156_LIM
## ROB-03: Kinematic Repeatability Error — Engineering Data Systems Pipeline
**Course:** Computer Programming | **Academic Year:** 2026

---

## Project Overview
This project applies the full automated Python data analytics pipeline to analyze **kinematic repeatability error** for the accuracy of industrial robots, this applies a statistical analysis and engineering-grade visualization on a filtered subset of a robot inverse kinematics dataset.

**Assigned Topic:** ROB-03 — Kinematic Repeatability Error
**Engineering Pillar:** Pillar 6 — Robotics & Mechatronics

---

## Repository Structure
```
EDS_TUPM-25-0156_Lim/
├── main.py                   # Full Python pipeline (OOP, 4 classes)
├── requirements.txt          # Required libraries
├── data/
│   ├── data_cleaned.csv # Filtered and cleaned dataset
│   └── data_original.csv # Raw Kaggle dataset
├── outputs/
│   ├── correlation_heatmap.png
│   ├── frame_analysis.avi
│   ├── temperature_vs_error.png
│   ├── trajectory.gif
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/migueljohnlim13-jpg/EDS_TUPM-25-0156_Lim.git
cd EDS_[StudentNum]_[Surname] # EDS_TUPM-25-0156_Lim (File Name)
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place the dataset
Copy `measurements_joint_configurations.csv` into the `data/` folder and rename it `data_original.csv`.

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
| `Visualizer` | 2 static image + 1 animation + 1 audio interleave (Matplotlib + Plotly) |
