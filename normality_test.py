import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


original_file = "TODO"
swapped_file  = "TODO"
# Load scores from each file (assumes one score per line)
scores_original = np.loadtxt(original_file)
scores_swapped  = np.loadtxt(swapped_file)


# --- compute differences ---
score_differences = scores_original - scores_swapped

score_differences = np.abs(score_differences)

# Shapiro–Wilk test for normality
w_stat, p_value = stats.shapiro(score_differences)
print(f"Shapiro–Wilk test: W = {w_stat:.4f}, p = {p_value:.4f}")
if p_value < 0.01:
    print("=> Reject normality (distribution is likely not normal)")
else:
    print("=> No evidence against normality (distribution is approximately normal)")