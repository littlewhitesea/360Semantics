import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import open_clip
from peft import PeftModel


def load_images_and_prompts(image_folder, prompt_file, preprocess):
    """
    Loads images and their corresponding text prompts.
    Assumes the Nth alphabetically sorted image corresponds to the Nth prompt.
    """
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    image_paths = sorted(Path(image_folder).glob("*.*"))
    images = [preprocess(Image.open(p).convert("RGB")) for p in tqdm(image_paths, desc="Preprocessing Images")]

    if len(images) != len(prompts):
        raise ValueError(f"Mismatch! Found {len(images)} images and {len(prompts)} prompts.")

    return torch.stack(images), prompts


def run_inference_on_folder(
        lora_model,
        tokenizer,
        preprocess,
        image_folder,
        prompt_file,
        output_path,
        device,
):
    """
    Runs the full inference pipeline for a given image folder and saves the scores.
    """
    print("\n" + "=" * 50)
    print(f"Processing folder: {image_folder}")
    print("=" * 50)

    # 1. Load your images & prompts using the model's preprocessor
    print("--- Loading and preprocessing data ---")
    image_input, prompts = load_images_and_prompts(image_folder, prompt_file, preprocess)

    # 2. Tokenize prompts with the same prefix used during training
    print("--- Tokenizing prompts ---")
    text_tokens = tokenizer(["This is " + desc for desc in prompts])

    # 3. Move tensors to the GPU
    image_input = image_input.to(device)
    text_tokens = text_tokens.to(device)

    # 4. Encode, normalize, and calculate scores
    print("--- Calculating scores ---")
    with torch.no_grad(), torch.autocast("cuda"):
        # Use the lora_model for the forward pass
        image_feats, text_feats, _ = lora_model(image_input, text_tokens)

        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)

        # Cosine similarity multiplied by 100
        scores = (image_feats * text_feats).sum(dim=-1).cpu().numpy() * 100

    print(f"Average fine-tuned CLIP score: {scores.mean():.4f}")

    # 5. Write scores to the output file
    with open(output_path, 'w') as f:
        for s in scores:
            # Save with high precision
            f.write(f"{s:.4f}\n")
    print(f"→ Scores saved successfully to: {output_path}")


def main():
    # Determine which device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Configuration Section ---
    ##########################################
    ##### TODO: Specify the paths below ######
    ##########################################
    # 1. List of paths to the folders containing your test images
    image_folders_to_process = [
        "TODO/laval_mixed_imgs_filtered",
        "TODO/laval_mixed_shift_128_imgs_filtered",
        "TODO/laval_mixed_shift_256_imgs_filtered",
        "TODO/laval_mixed_shift_384_imgs_filtered",
        "TODO/laval_mixed_shift_512_imgs_filtered",
        "TODO/laval_mixed_shift_640_imgs_filtered",
        "TODO/laval_mixed_shift_768_imgs_filtered",
        "TODO/laval_mixed_shift_896_imgs_filtered",
    ]

    # image_folders_to_process = [
    #     "TODO/diffusion360_mixed_imgs_flitered",
    #     "TODO/diffusion360_mixed_shift_128_imgs_flitered",
    #     "TODO/diffusion360_mixed_shift_256_imgs_flitered",
    #     "TODO/diffusion360_mixed_shift_384_imgs_flitered",
    #     "TODO/diffusion360_mixed_shift_512_imgs_flitered",
    #     "TODO/diffusion360_mixed_shift_640_imgs_flitered",
    #     "TODO/diffusion360_mixed_shift_768_imgs_flitered",
    #     "TODO/diffusion360_mixed_shift_896_imgs_flitered",
    # ]

    # 2. Path to the text file containing your test prompts (one per line)
    #    (This is assumed to be the same for all image folders)
    prompt_file = "TODO"

    # 3. Path to the SPECIFIC LoRA adapter checkpoint you want to use
    adapter_checkpoint_path = "TODO"

    # 4. Directory where the output score files will be saved
    output_dir = Path("TODO")
    #########################################
    #########################################

    # --- These settings MUST match your training script ---
    MODEL_NAME = 'ViT-B-32'
    PRETRAINED_DATASET = 'laion400m_e32'

    # Create the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n=== Running Inference with Fine-Tuned LoRA Model ===")

    # 1. Load the base model, preprocessor, and tokenizer (only once)
    print(f"--- Loading base model: {MODEL_NAME} ({PRETRAINED_DATASET}) ---")
    base_model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED_DATASET
    )
    base_model.to(device)
    base_model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    # 2. Load the LoRA adapter and apply it to the base model (only once)
    print(f"--- Loading LoRA adapter from: {adapter_checkpoint_path} ---")
    try:
        base_model.visual = PeftModel.from_pretrained(base_model.visual, adapter_checkpoint_path)
        lora_model = base_model
        # lora_model = PeftModel.from_pretrained(base_model, adapter_checkpoint_path)
        lora_model.to(device)
        lora_model.eval()
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Please ensure the `adapter_checkpoint_path` is correct.")
        return

    # 3. Loop through each folder, generate names, and run inference
    for image_folder in image_folders_to_process:
        # Dynamically create the output filename based on the image folder name
        folder_name = Path(image_folder).name

        # This logic extracts the "shift_XXX" part, or nothing if it's the base folder
        suffix = folder_name.replace("laval_mixed", "").replace("_imgs", "")
        output_fname = f"lora_finetuned{suffix}_scores.txt"
        output_path = output_dir / output_fname

        run_inference_on_folder(
            lora_model=lora_model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            image_folder=image_folder,
            prompt_file=prompt_file,
            output_path=output_path,
            device=device,
        )

    print("\n=== All processing complete. ===")


if __name__ == "__main__":
    main()