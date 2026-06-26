"""
config.py
"""
import os
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT   = PROJECT_DIR.parent / "Data Set"
OUTPUT_ROOT = PROJECT_DIR / "Output"

# Sibling source folders inside DATA_ROOT (one file per part in each).
SOURCE_DIRS = {
    "nominal": DATA_ROOT / "Nominal STL Files",
    "ct":      DATA_ROOT / "actual CT scan",
    "zephyr":  DATA_ROOT / "Zephyr STL",
}
PART_SIZE_RE = r"(?<!\d)([246])\s*(?:mm|P)"   
SIZE_TO_PARTID = {"2": "2PR", "4": "4P2", "6": "6P1"}  
SCALE_AUTODETECT_TOL = 0.05        
SCALE_OVERRIDE = {                   
 
}

SAMPLE_METHOD = os.environ.get("SAMPLE_METHOD", "poisson") 
POISSON_SAMPLES = {
    "nominal": 60000,
    "ct":      120000,
    "zephyr":  120000,
}
NORMAL_RADIUS_FACTOR = 3.0       
NORMAL_MAX_NN = 30

# Phase 2 -- registration (Nominal is the fixed reference frame)

VOXEL_SIZE = 0.4              
FPFH_RADIUS_FACTOR = 5.0            
RANSAC_DIST_FACTOR = 1.5          
RANSAC_N = 4                         
RANSAC_MAX_ITER = 4_000_000
RANSAC_CONFIDENCE = 0.999
RANSAC_SEED = 0
RANSAC_SEEDS = (0, 1, 2, 3)
REG_COVERAGE_TOL = 0.5            
ICP_DIST_FACTOR = 2.0              
ICP_MAX_ITER = 100
LEG_REFINE = True
LEG_REFINE_DIST_FACTOR = 1.5
UPRIGHT_REFINE = True
UPRIGHT_MAX_ROLL_DEG = 3.0
UPRIGHT_MAX_APPLY_DEG = 20.0


# Phase 3 -- surface deviation

DEVIATION_PAIRS = [                  
    ("ct", "nominal"),
    ("zephyr", "nominal"),
    ("ct", "zephyr"),
]
HEATMAP_CLIP = 1.0                  
HIST_BINS = 80


# Phase 4 -- segmentation (legs + overhang) in the aligned Nominal frame

END_SLICE_FRAC = 0.15           
BIMODAL_BINS = 24              
WALL_BAND_FRAC = 0.30               
OVERHANG_W_FRAC = 0.80               
OVERHANG_V_FRAC = 0.12               
DBSCAN_EPS_FACTOR = 3.0              
DBSCAN_MIN_POINTS = 20
PLANE_DIST_THRESH = 0.20           
PLANE_RANSAC_N = 3
PLANE_NUM_ITER = 2000


# Phase 5 -- measurement

OVERHANG_MAX_TILT_DEG = 30.0
NOMINAL_LEG_ANGLE_DEG = 90.0

# Phase 6 -- reporting

REGISTRATION_CSV = OUTPUT_ROOT / "registration_report.csv"
COMPARISON_CSV   = OUTPUT_ROOT / "comparison_report.csv"
DISTORTION_CSV   = OUTPUT_ROOT / "distortion_report.csv"

# Colours for segment-coloured clouds / annotated plots
SEG_COLORS = {
    "leg_1":            (0.85, 0.20, 0.20),
    "leg_2":            (0.20, 0.40, 0.85),
    "overhang_surface": (0.20, 0.75, 0.30),
    "other":            (0.70, 0.70, 0.70),
}

SOURCES = ("nominal", "ct", "zephyr")
