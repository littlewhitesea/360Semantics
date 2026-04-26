import numpy as np
from scipy import stats
import os
import glob
import re

# --- Define Base Paths and Output File ---
folder_path = "TODO"
model_name = "mixed_scores_ViT_B_32.txt"
original_file = os.path.join(folder_path, model_name)
output_results_file = os.path.join(folder_path, "TODO")

# --- MODIFIED: Define Parameters for the One-Sided Stability Test ---
# Alpha is the significance level.
alpha = 0.01
# Stability bound (β, beta).
stability_bound = 1.7919

# --- 1. Automatically Find Transformed Files ---
base_dir = os.path.dirname(os.path.dirname(original_file))
filename = os.path.basename(original_file)
glob_pattern = os.path.join(base_dir, 'TODO', filename)

print(f"Searching for transformed files using pattern: {glob_pattern}\n")
transformed_files = glob.glob(glob_pattern)


def get_shift_number(path):
    match = re.search(r'shift_(\d+)_imgs_scores', path)
    return int(match.group(1)) if match else 0


transformed_files.sort(key=get_shift_number)

if not transformed_files:
    print("FATAL ERROR: No transformed files found matching the pattern. Exiting.")
    exit()

print("Found the following transformed files to test:")
for f in transformed_files:
    print(f"  - {f}")
print("-" * 50)

# --- 2. Load Original Scores (once) ---
try:
    scores_original = np.loadtxt(original_file)
    print(f"Successfully loaded original scores from: {os.path.basename(original_file)}\n")
except FileNotFoundError:
    print(f"FATAL ERROR: Original file not found at {original_file}. Cannot proceed.")
    exit()

# --- 3. Loop, Test, and Save Results to File ---
with open(output_results_file, 'w') as f_out:
    # MODIFIED: Write a new header for the stability test results file
    f_out.write("One-Sided Wilcoxon Signed-Rank Test for Score Stability\n")
    f_out.write(f"Stability Bound (β): {stability_bound}\n")
    f_out.write("Null Hypothesis (H0): The change is NOT small. median(|Original - Transformed|) >= β\n")
    f_out.write(
        "Alternative Hypothesis (H1): The change IS small (scores are stable). median(|Original - Transformed|) < β\n")
    f_out.write(f"Alpha level for significance: {alpha}\n")
    f_out.write(f"Original (A) File: {original_file}\n\n")

    for transformed_file in transformed_files:
        separator = "=" * 60
        transformation_name = os.path.basename(os.path.dirname(transformed_file))
        header = f"{separator}\nTESTING AGAINST: {transformation_name}\n{separator}\n"

        print(header, end='')
        f_out.write(header)

        try:
            scores_transformed = np.loadtxt(transformed_file)
        except (FileNotFoundError, Exception) as e:
            error_msg = f"  -> ERROR: Could not load or process file. Error: {e}. SKIPPING.\n\n"
            print(error_msg)
            f_out.write(error_msg)
            continue

        if scores_original.shape != scores_transformed.shape:
            error_msg = f"  -> WARNING: Shape mismatch ({scores_original.shape} vs {scores_transformed.shape}). SKIPPING.\n\n"
            print(error_msg)
            f_out.write(error_msg)
            continue

        A = scores_original
        B = scores_transformed

        abs_diff = np.abs(A - B)

        try:
            stat, p_value = stats.wilcoxon(abs_diff - stability_bound, alternative='less')
        except ValueError as e:
            # This can happen if all differences are equal to the bound, which is very unlikely.
            error_msg = f"  -> STATISTICAL ERROR: Wilcoxon test failed. Error: {e}. SKIPPING.\n\n"
            print(error_msg)
            f_out.write(error_msg)
            continue

        context_str = (
            f"n = {abs_diff.size}, "
            f"median(|A-B|) = {np.median(abs_diff):.4f}, "
            f"mean(|A-B|) = {abs_diff.mean():.4f}\n"
        )

        test_result_str = (
            f"One-sided test (H1: median(|diff|) < {stability_bound}): stat = {stat:.4f}, p-value = {p_value:.4f}\n"
        )

        # MODIFIED: The interpretation is now about stability, not equivalence.
        if p_value < alpha:
            conclusion_str = (
                f"Conclusion: Reject H0. The evidence suggests the scores are STABLE, as the median absolute\n"
                f"            difference is significantly smaller than the bound of {stability_bound}.\n"
            )
        else:
            conclusion_str = (
                f"Conclusion: Fail to reject H0. Insufficient evidence to claim stability. The median absolute\n"
                f"            difference is not significantly smaller than the bound of {stability_bound}.\n"
            )

        # Print to console
        print(context_str, end='')
        print(test_result_str, end='')
        print(conclusion_str)

        # Write to file
        f_out.write(context_str)
        f_out.write(test_result_str)
        f_out.write(conclusion_str + "\n")

print(f"\n{'-' * 50}\nProcessing complete. Stability test results saved to '{output_results_file}'\n")