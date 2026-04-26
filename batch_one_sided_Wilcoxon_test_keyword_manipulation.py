import numpy as np
from scipy import stats
import os


def run_wilcoxon_tests():
    """
    Loads pairs of score files, performs a one-sided Wilcoxon signed-rank test,
    and saves the results to a summary text file.
    """
    # ——— paths ———
    ##########################################
    ##### TODO ###############################
    ##### Specify the path of following txt files
    ##########################################
    original_folder = "TODO"
    #########################################
    #########################################
    #########################################

    # Base path
    output_base = "TODO"

    # Loop through the four variants
    for i in ["first", "second", "third", "fourth"]:
        transformed_folder = f"{output_base}/laion400m_imgs_scores_{i}_generic"

        print("Folder:", transformed_folder)

        # --- Configuration ---
        file_names = [
            "mixed_scores_ViT_B_32.txt",
            "mixed_scores_ViT_B_16.txt",
            "mixed_scores_ViT_L_14.txt"
        ]

        # Define the output file path
        output_file_path = os.path.join(transformed_folder, "wilcoxon_test_results_third_format.txt")

        # Significance level for interpretation
        ALPHA = 0.01

        print(f"Starting Wilcoxon tests. Results will be saved to:\n{output_file_path}\n")

        # Use a 'with' statement to safely open and write to the output file
        try:
            with open(output_file_path, 'w') as f_out:
                # Write a header to the output file for clarity
                f_out.write("One-Sided Wilcoxon Signed-Rank Test Results (Keyword Manipulation)\n")
                f_out.write("============================================\n\n")
                f_out.write(
                    "This file contains the results of a one-sided Wilcoxon test comparing 'original' vs. 'transformed' scores.\n")
                f_out.write(
                    "The alternative hypothesis (Hₐ) is that the original scores are significantly GREATER than the transformed scores.\n")
                f_out.write(f"Significance Level (alpha) used for interpretation: {ALPHA}\n\n")

                # Iterate directly over the file names for cleaner code
                for file_name in file_names:
                    print(f"Processing: {file_name}...")

                    original_file = os.path.join(original_folder, file_name)
                    transformed_file = os.path.join(transformed_folder, file_name)

                    # Write the header for this specific test to the file
                    f_out.write(f"--- Test for: {file_name} ---\n")

                    try:
                        # Load data from text files
                        scores_original = np.loadtxt(original_file)
                        scores_transformed = np.loadtxt(transformed_file)

                        # Check that both arrays have the same length
                        if scores_original.shape != scores_transformed.shape:
                            error_msg = "Error: The two score files do not have the same number of entries. Skipping."
                            print(error_msg)
                            f_out.write(f"{error_msg}\n\n")
                            continue

                        # --- One-sided Wilcoxon signed-rank test Section ---
                        # H₀: The median of the differences is zero.
                        # Hₐ: The median of the differences (original - transformed) is greater than zero.
                        stat, p_val = stats.wilcoxon(
                            scores_original,
                            scores_transformed,
                            zero_method='wilcox',
                            alternative='greater'
                        )

                        # --- Write Results to File ---
                        f_out.write(f"  Statistic (W⁺) = {stat:.4f}\n")
                        f_out.write(f"  p-value        = {p_val:.4f}\n")

                        # Interpretation
                        if p_val < ALPHA:
                            result_text = f"Result: Reject H₀. Evidence suggests original scores are significantly greater than transformed scores."
                        else:
                            result_text = f"Result: Fail to reject H₀. No significant evidence that original scores exceed transformed scores."

                        f_out.write(f"  {result_text}\n\n")

                    except FileNotFoundError:
                        error_msg = f"Error: Could not find one or both files for '{file_name}'. Please check paths. Skipping."
                        print(error_msg)
                        f_out.write(f"{error_msg}\n\n")
                    except Exception as e:
                        error_msg = f"An unexpected error occurred while processing {file_name}: {e}"
                        print(error_msg)
                        f_out.write(f"{error_msg}\n\n")

            print("Processing complete. All results have been saved successfully.")

        except IOError as e:
            print(
                f"Error: Could not write to output file at '{output_file_path}'.\nPlease check permissions and if the directory exists. Details: {e}")


# Standard Python entry point
if __name__ == "__main__":
    run_wilcoxon_tests()