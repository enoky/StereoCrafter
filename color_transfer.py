import os
import cv2
import torch
import numpy as np


def ctsot_pytorch(src, trg, steps=10, batch_size=5):
    """
    Color Transfer via Sliced Optimal Transport (SOT) using PyTorch for GPU Acceleration.

    Parameters:
        src (numpy.ndarray): Source image (H, W, C), float32, range [0,1].
        trg (numpy.ndarray): Target image (H, W, C), float32, range [0,1].
        steps (int): Number of solver steps.
        batch_size (int): Solver batch size.

    Returns:
        numpy.ndarray: Color-transferred image (float32, range [0,1]).
    """
    device = torch.device("cuda")  # Use GPU if available

    src = torch.tensor(src, dtype=torch.float32, device=device)
    trg = torch.tensor(trg, dtype=torch.float32, device=device)

    h, w, c = src.shape
    new_src = src.clone()

    for step in range(steps):
        advect = torch.zeros((h * w, c), dtype=torch.float32, device=device)
        for batch in range(batch_size):
            dir = torch.randn(c, dtype=torch.float32, device=device)
            dir /= torch.norm(dir) + 1e-8  # Normalize direction vector

            projsource = torch.sum(new_src * dir, dim=-1).reshape(h * w)
            projtarget = torch.sum(trg * dir, dim=-1).reshape(h * w)

            idSource = torch.argsort(projsource)
            idTarget = torch.argsort(projtarget)

            a = projtarget[idTarget] - projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += a * dir[i_c]

        new_src += advect.reshape((h, w, c)) / batch_size

    new_src = torch.clamp(new_src, 0, 1)  # Clamp to [0,1] without converting to uint8
    return new_src.cpu().numpy()  # Convert back to NumPy


def batch_process(source_folder, target_folder, output_folder):
    """
    Batch process images using GPU-accelerated color transfer.

    Parameters:
        source_folder (str): Path to the folder containing source images.
        target_folder (str): Path to the folder containing target images.
        output_folder (str): Path to save the output images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    source_images = sorted([f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    target_images = sorted([f for f in os.listdir(target_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(source_images) != len(target_images):
        print("⚠️ Warning: Different numbers of source and target images.")

    for src_name, trg_name in zip(source_images, target_images):
        src_path = os.path.join(source_folder, src_name)
        trg_path = os.path.join(target_folder, trg_name)
        output_path = os.path.join(output_folder, src_name)

        print(f"Processing: {src_name} ⟶ {trg_name}")

        # Read images
        src = cv2.imread(src_path).astype(np.float32) / 255.0  # Normalize to [0,1]
        trg = cv2.imread(trg_path).astype(np.float32) / 255.0

        if src is None or trg is None:
            print(f"❌ Error loading {src_name} or {trg_name}, skipping...")
            continue

        # Resize target image to match source dimensions if necessary
        if src.shape[:2] != trg.shape[:2]:
            trg = cv2.resize(trg, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # Apply GPU-accelerated color transfer
        transferred = ctsot_pytorch(src, trg)

        # Save the result with high quality
        transferred_uint16 = (transferred * 65535).astype(np.uint16)  # Convert to 16-bit
        cv2.imwrite(output_path, transferred_uint16, [cv2.IMWRITE_PNG_COMPRESSION, 3])  # balanced compression
        print(f"✅ Saved: {output_path}")


# Define folders
source_folder = "input/source"  # inpainted PNG frames
target_folder = "input/target"  # original source PNG frames
output_folder = "output"

# Run batch processing
batch_process(source_folder, target_folder, output_folder)