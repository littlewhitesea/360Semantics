# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
import random
import numpy as np


import open_clip
from peft import get_peft_model, LoraConfig

import pdb

# --- Configuration ---

# Set a seed for reproducibility
SEED = 0

IMAGE_DATA_PATH = "TODO"
PROMPTS_FILE_PATH = "TODO"
SCORES_FILE_PATH = "TODO"

# This block contains all the settings for the training run.
# Make sure these match in your inference script!
MODEL_NAME = 'ViT-B-32'
PRETRAINED_DATASET = 'laion400m_e32'
LORA_ADAPTER_DIR = "TODO"  # Directory to save the LoRA adapter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

### Define the path for the loss log file
LOSS_LOG_FILE = os.path.join(LORA_ADAPTER_DIR, "training_loss_log.txt")
### Define the path for the checkpoint file used for resuming training
CHECKPOINT_FILE = os.path.join(LORA_ADAPTER_DIR, "latest_checkpoint.pth")

os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
GRADIENT_CLIP_VAL = 1.0

LAMBDA_SHIFT = 1
SAVE_EVERY_N_EPOCHS = 2

### Function to set seed for reproducibility
def set_seed(seed_value):
    """Sets the seed for reproducibility in PyTorch, NumPy, and Python's random module."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed_value}")


# Helper function to print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


# ### Custom Charbonnier Loss Function

class CharbonnierLoss(nn.Module):
    """The Charbonnier loss function (a smooth L1 loss)."""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, predicted, target):
        loss = torch.sqrt((predicted - target)**2 + self.eps**2)
        return torch.mean(loss)


def horizontally_circular_shift_numpy(image: Image.Image) -> Image.Image:
    """
    Performs a horizontal circular shift on a PIL Image using NumPy.
    """
    # 1. Convert PIL Image to NumPy array. The shape is (Height, Width, Channels)
    image_np = np.array(image)

    # 2. Get width and determine a random shift amount
    width = image_np.shape[1]
    shift = random.randint(0, width - 1)

    # 3. Perform the circular shift along the width axis (axis=1)
    # This is the core operation.
    shifted_np = np.roll(image_np, shift=shift, axis=1)

    # 4. Convert the shifted NumPy array back to a PIL Image
    shifted_image = Image.fromarray(shifted_np)

    return shifted_image


class ImageCaptionScoreDataset(Dataset):
    """
    A custom dataset class that loads images and their corresponding captions
    from a directory. It assumes that for each `image_name.jpg`, there is a
    corresponding `image_name_caption.txt`.
    """

    def __init__(self, image_dir, prompts_file, scores_file, preprocess, tokenizer):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts file not found at: {prompts_file}")

        try:
            with open(scores_file, 'r') as f:
                self.scores = [float(line.strip()) for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Scores file not found at: {scores_file}")

        # --- CRITICAL SANITY CHECK ---
        # Ensure all three lists are the same length.
        num_images = len(self.image_paths)
        num_prompts = len(self.prompts)
        num_scores = len(self.scores)

        if not (num_images == num_prompts == num_scores):
            raise ValueError(
                "CRITICAL ERROR: Data source length mismatch! "
                f"Found {num_images} images, {num_prompts} prompts, and {num_scores} scores. "
                "All three must be the same. Please check your files."
            )

        if num_images == 0:
            raise ValueError("No images found. Please check the image_dir path.")

    def __len__(self):
        """Returns the total number of image-caption pairs."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and returns a single image-caption pair from the dataset.
        """
        # Get the parallel data for the given index
        image_path = self.image_paths[idx]
        caption = self.prompts[idx]
        score = self.scores[idx]

        modified_caption = f"This is {caption}"

        image = Image.open(image_path).convert("RGB")

        # Create the shifted PIL image using the NumPy method
        shifted_image = horizontally_circular_shift_numpy(image)

        # Apply the CLIP preprocessor to both images
        original_image_tensor = self.preprocess(image)
        shifted_image_tensor = self.preprocess(shifted_image)

        text_tensor = self.tokenizer(modified_caption)[0]
        score_tensor = torch.tensor(score, dtype=torch.float)

        return original_image_tensor, shifted_image_tensor, text_tensor, score_tensor


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

### Set the seed
set_seed(SEED)

# --- 1. Load Base Model, Preprocessor, and Tokenizer ---
print(f"Using device: {DEVICE}")
print(f"Loading base model: {MODEL_NAME} ({PRETRAINED_DATASET})")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED_DATASET,
    device=DEVICE
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# --- 2. Configure and Apply LoRA ---
print("Configuring and applying LoRA to the IMAGE ENCODER ONLY...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["in_proj", "out_proj", "c_fc"],
    lora_dropout=0.1,
    bias="none",
)


# Freeze original model parameters and apply LoRA
model.requires_grad_(False)
print("Injecting LoRA adapters into the Vision Transformer...")
model.visual = get_peft_model(model.visual, lora_config)
lora_model = model
print("\nModel architecture with LoRA adapters (Image Encoder Only):")
print_trainable_parameters(lora_model)


# --- 3. Instantiate Custom Dataset and DataLoader ---
print(f"\nLoading images from: {IMAGE_DATA_PATH}")
print(f"Loading prompts from: {PROMPTS_FILE_PATH}")
print(f"Loading scores from: {SCORES_FILE_PATH}")

dataset = ImageCaptionScoreDataset(
    image_dir=IMAGE_DATA_PATH,
    prompts_file=PROMPTS_FILE_PATH,
    scores_file=SCORES_FILE_PATH,
    preprocess=preprocess,
    tokenizer=tokenizer
)

# --- IMPORTANT: Use shuffle=True for real training ---
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True # Shuffles data at every epoch
)


# --- 4. Define Loss and Training Loop ---

loss_fn = CharbonnierLoss(eps=1e-3) # You can tune the epsilon value here
optimizer = optim.AdamW(lora_model.parameters(), lr=LEARNING_RATE)


start_epoch = 0
if os.path.exists(CHECKPOINT_FILE):
    print(f"--- Resuming training from checkpoint: {CHECKPOINT_FILE} ---")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    lora_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    print(f"--- Resumed from Epoch {checkpoint['epoch'] + 1}. Starting next at Epoch {start_epoch + 1} ---")
else:
    print("--- No checkpoint found. Starting training from scratch. ---")
    # If starting fresh, ensure log file is new
    with open(LOSS_LOG_FILE, 'w') as f:
        f.write("Epoch,Average_Loss\n")

print("\n--- Starting Training ---")
# os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)

lora_model.train()
for epoch in range(start_epoch, NUM_EPOCHS):
    total_epoch_loss = 0
    for i, (images, shifted_images, texts, ground_truth_scores) in enumerate(dataloader):
        images = images.to(DEVICE)
        shifted_images = shifted_images.to(DEVICE)
        texts = texts.to(DEVICE)
        ground_truth_scores = ground_truth_scores.to(DEVICE)

        optimizer.zero_grad()

        b_size = images.shape[0]
        combined_images = torch.cat([images, shifted_images], dim=0)
        combined_texts = texts.repeat(2, 1)

        image_features_combined, text_features_combined, _ = lora_model(combined_images, combined_texts)

        # Split results back into original and shifted components
        original_image_features = image_features_combined[:b_size]
        shifted_image_features = image_features_combined[b_size:]
        text_features = text_features_combined[:b_size]

        # Normalize embeddings for cosine similarity calculation
        original_image_features = F.normalize(original_image_features, dim=-1)
        shifted_image_features = F.normalize(shifted_image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Calculate predicted scores for both sets
        predicted_scores_original = (original_image_features * text_features).sum(dim=1) * 100
        predicted_scores_shifted = (shifted_image_features * text_features).sum(dim=1) * 100

        # pdb.set_trace()

        loss_predicted = loss_fn(predicted_scores_shifted, predicted_scores_original)
        loss_shifted = loss_fn(predicted_scores_shifted, ground_truth_scores)
        total_loss = LAMBDA_SHIFT * loss_predicted + (1 - LAMBDA_SHIFT) * loss_shifted


        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_model.parameters(), GRADIENT_CLIP_VAL)
        optimizer.step()

        total_epoch_loss += total_loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{i + 1}/{len(dataloader)}], "
                  f"Total Loss: {total_loss.item():.4f} (Predicted: {loss_predicted.item():.4f}, Shifted: {loss_shifted.item():.4f})")

    avg_epoch_loss = total_epoch_loss / len(dataloader)
    print(f"--- End of Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4f} ---")
    with open(LOSS_LOG_FILE, 'a') as f:
        f.write(f"{epoch + 1},{avg_epoch_loss:.4f}\n")

    if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
        # 1. Save the consolidated checkpoint for resuming
        print(f"--- Saving consolidated checkpoint to {CHECKPOINT_FILE} ---")
        torch.save({
            'epoch': epoch,
            'model_state_dict': lora_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, CHECKPOINT_FILE)

        epoch_save_path = os.path.join(LORA_ADAPTER_DIR, f"epoch_{epoch + 1}")
        print(f"--- Saving LoRA adapter for epoch {epoch + 1} to {epoch_save_path} ---")
        lora_model.visual.save_pretrained(epoch_save_path)

print("--- Training Complete ---")

# --- 5. Save the LoRA Adapter ---
final_save_path = os.path.join(LORA_ADAPTER_DIR, "final")
print(f"\nSaving final LoRA adapter to {final_save_path}...")
lora_model.visual.save_pretrained(final_save_path)
print("Adapter saved successfully.")