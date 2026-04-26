import numpy as np
import torch
from pathlib import Path
import open_clip
from open_clip import tokenizer
from PIL import Image, UnidentifiedImageError # Import the specific exception

import pdb

def load_images_and_prompts(image_folder, prompt_file, preprocess):
    """Load images and their corresponding text prompts."""
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]

    images = []
    for image_path in sorted(Path(image_folder).rglob("*.*")):
        image = Image.open(image_path).convert("RGB")
        images.append(preprocess(image))

    return torch.stack(images), prompts


def main():

    # Determine which device to use
    device = "cuda" if torch.cuda.is_available() else "cpu" # <-- ADDED FOR GPU

    # ——— paths ———
    ##########################################
    ##### TODO ###############################
    ##### Specify the path of following folders or file
    ##########################################
    image_folder = "TODO"
    prompt_file  = "TODO"
    output_dir   = Path("TODO")
    #########################################
    #########################################
    #########################################


    output_dir.mkdir(exist_ok=True, parents=True)

    # ——— loop over models ———

    ###### OpenCLIP Trained on LAION-400M
    for model_name in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:

        print(f"\n=== {model_name} ===")

        # Determine the correct pretrained tag based on the model name
        if model_name == "ViT-B-32":
            pretrained_tag = "laion400m_e32"
        elif model_name == "ViT-B-16":
            pretrained_tag = "laion400m_e32"
        elif model_name == "ViT-L-14":
            pretrained_tag = "laion400m_e32"
        else:
            pretrained_tag = "nothing"

    ###### OpenCLIP Trained on LAION-2B
    # for model_name in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
    #
    #     print(f"\n=== {model_name} ===")
    #
    #     # Determine the correct pretrained tag based on the model name
    #     if model_name == "ViT-B-32":
    #         pretrained_tag = "laion2b_s34b_b79k"
    #     elif model_name == "ViT-B-16":
    #         pretrained_tag = "laion2b_s34b_b88k"
    #     elif model_name == "ViT-L-14":
    #         pretrained_tag = "laion2b_s32b_b82k"
    #     else:
    #         pretrained_tag = "nothing"

    ###### OpenAI
    # for model_name in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
    #
    #     pretrained_tag = "openai"

        print(f"--- Using pretrained tag: {pretrained_tag} ---")

        # recreate model+preprocess for this variant
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
        model.to(device) # <-- ADDED FOR GPU
        model.eval()

        # reload your images & prompts with this model’s preprocess
        image_input, prompts = load_images_and_prompts(image_folder, prompt_file, preprocess)

        # tokenize once
        # text_tokens = tokenizer.tokenize(prompts)
        text_tokens = tokenizer.tokenize(["This is " + desc for desc in prompts])

        # Move tensors to the GPU
        image_input = image_input.to(device) # <-- ADDED FOR GPU
        text_tokens = text_tokens.to(device) # <-- ADDED FOR GPU

        # encode & normalize
        with torch.no_grad(), torch.autocast("cuda"):
            image_feats = model.encode_image(image_input).float()
            text_feats  = model.encode_text(text_tokens).float()
        image_feats /= image_feats.norm(dim=-1, keepdim=True)
        text_feats  /= text_feats.norm(dim=-1, keepdim=True)

        # cosine similarity ×100
        scores = (image_feats * text_feats).sum(dim=-1).cpu().numpy() * 100
        print(f"Average CLIP score: {scores.mean():.4f}")

        # pdb.set_trace()

        #########################################
        ##### TODO ##############################
        ##### Specify the name of the saved txt file
        #########################################
        # write to file
        fname = f"mixed_scores_{model_name.replace('-', '_')}.txt"
        #########################################
        #########################################
        #########################################



        outpath = output_dir / fname
        with open(outpath, 'w') as f:
            for s in scores:
                f.write(f"{s:.4f}\n")
        print(f"→ saved: {outpath}")

if __name__ == "__main__":
    main()