import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ——— paths ———
##########################################
##### TODO ###############################
##### You only need to specify the path to the main scores file.
##### The script will find the others and save the results in the same folder.
##########################################
original_file_path = "TODO"

# --- Stability Test Configuration ---

# Stability bound (beta).
stability_bound = 1.7919

# Significance level (alpha)
alpha = 0.01

# --- Configuration (derived automatically) ---

# Define the filenames of the other score files you want to test against
comparison_filenames = [
    "lora_finetuned_shift_128_filtered_scores.txt",
    "lora_finetuned_shift_256_filtered_scores.txt",
    "lora_finetuned_shift_384_filtered_scores.txt",
    "lora_finetuned_shift_512_filtered_scores.txt",
    "lora_finetuned_shift_640_filtered_scores.txt",
    "lora_finetuned_shift_768_filtered_scores.txt",
    "lora_finetuned_shift_896_filtered_scores.txt",
]

# Use pathlib to intelligently handle paths
original_path_obj = Path(original_file_path)
results_dir = original_path_obj.parent
output_results_file = results_dir / "all_wilcoxon_stability_test_results.txt"

# --- Main Logic ---

# Check if the original file exists before proceeding
if not original_path_obj.is_file():
    print(f"FATAL ERROR: The original file was not found at: {original_path_obj}")
    exit()

# Load the original scores once
print(f"Loading original scores from: {original_path_obj.name}")
scores_original = np.loadtxt(original_path_obj)

# Open the output file in 'write' mode.
print(f"Results will be saved to: {output_results_file}")
with open(output_results_file, 'w', encoding='utf-8') as f_out:
    f_out.write("One-Sided Wilcoxon Signed-Rank Test for Score Stability\n")
    f_out.write("=" * 60 + "\n")
    f_out.write(f"Stability Bound (β): {stability_bound}\n")
    f_out.write(f"Significance Level (α): {alpha}\n")
    f_out.write("Null Hypothesis (H0): The model is NOT stable. median(|Original - Transformed|) >= β\n")
    f_out.write("Alternative Hypothesis (H1): The model IS stable. median(|Original - Transformed|) < β\n")
    f_out.write("=" * 60 + "\n\n")

    # Loop through each of the comparison filenames
    for filename in comparison_filenames:
        # Construct the full, absolute path for the current transformed file
        transformed_file_path = results_dir / filename

        print(f"\n--- Comparing with: {filename} ---")
        f_out.write(f"--- Comparison: '{original_path_obj.name}' vs. '{filename}' ---\n")

        # First, check if the file we want to compare against actually exists
        if not transformed_file_path.is_file():
            error_msg = f"Error: The file '{filename}' was not found in the directory. Skipping."
            print(error_msg)
            f_out.write(error_msg + "\n\n")
            continue

        try:
            # Load scores for the current transformed file
            scores_transformed = np.loadtxt(transformed_file_path)

            # Check that both arrays have the same length
            if scores_original.shape != scores_transformed.shape:
                error_msg = "ValueError: The two score files do not have the same number of entries."
                print(error_msg)
                f_out.write(error_msg + "\n\n")
                continue

            # --- MODIFIED: One-Sided Stability Test Section ---

            # 1. Calculate the absolute differences, which is the data we are testing.
            abs_diff = np.abs(scores_original - scores_transformed)

            # 2. Perform the one-sided Wilcoxon test.
            # H0: median(abs_diff) >= stability_bound
            # H1: median(abs_diff) < stability_bound
            # This is statistically equivalent to testing if median(abs_diff - stability_bound) < 0
            stat, p_val = stats.wilcoxon(abs_diff - stability_bound, alternative='less')

            # 3. Create a more informative result string
            # It's useful to see the median absolute difference to understand the result.
            context_string = (
                f"Median absolute difference = {np.median(abs_diff):.4f} (Bound β = {stability_bound})\n"
                f"Test results: statistic = {stat:.4f}, p-value = {p_val:.4f}\n"
            )

            # 4. Update the interpretation to reflect the stability test
            if p_val < alpha:
                interpretation = "Conclusion: Reject H0. Evidence suggests the scores are STABLE (median change is significantly less than the bound)."
            else:
                interpretation = "Conclusion: Fail to reject H0. Insufficient evidence to claim stability (median change is not significantly less than the bound)."

            result_string = context_string + interpretation

            print(result_string)
            f_out.write(result_string + "\n\n")

        except Exception as e:
            error_msg = f"An unexpected error occurred while processing '{filename}': {e}"
            print(error_msg)
            f_out.write(error_msg + "\n\n")

print(f"\nAll tests complete. Results have been saved to '{output_results_file}'.")