
# ==========================================
# metrics.py
# Image Quality and Segmentation Evaluation
# MSE, PSNR, IoU
# ==========================================

import cv2
import numpy as np

# Support package and direct execution
if __package__:
    from .data import (
        get_image_mask_pairs,
        load_image,
        load_ground_truth,
        degrade_image
    )
else:
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.data import (
        get_image_mask_pairs,
        load_image,
        load_ground_truth,
        degrade_image
    )


# ==========================================
# Cell 1: Mean Squared Error (MSE)
# ==========================================

def calculate_mse(original, processed):

    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)

    if original.shape != processed.shape:
        raise ValueError("Images must have the same shape")

    mse = np.mean((original - processed) ** 2)

    return float(mse)


# ==========================================
# Cell 2: Peak Signal-to-Noise Ratio (PSNR)
# ==========================================

def calculate_psnr(original, processed, max_pixel=255):

    mse = calculate_mse(original, processed)

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10((max_pixel ** 2) / mse)

    return float(psnr)


# ==========================================
# Cell 3: Intersection over Union (IoU)
# ==========================================

def calculate_iou(predicted_mask, ground_truth_mask):

    predicted_mask = np.asarray(predicted_mask) > 0
    ground_truth_mask = np.asarray(ground_truth_mask) > 0

    if predicted_mask.shape != ground_truth_mask.shape:
        raise ValueError("Masks must have the same shape")

    intersection = np.logical_and(
        predicted_mask,
        ground_truth_mask
    ).sum()

    union = np.logical_or(
        predicted_mask,
        ground_truth_mask
    ).sum()

    if union == 0:
        return 1.0

    iou = intersection / union

    return float(iou)


# ==========================================
# Cell 4: Hand-Crafted Tests
# ==========================================

def run_hand_crafted_tests():

    print("=" * 50)
    print("Running Hand-Crafted Tests")
    print("=" * 50)

    # MSE Test
    original = np.array([0, 10, 20, 30])
    processed = np.array([0, 12, 23, 34])

    expected_mse = 7.25
    computed_mse = calculate_mse(original, processed)

    print("\nMSE Test")
    print("Expected MSE:", expected_mse)
    print("Computed MSE:", computed_mse)

    assert np.isclose(computed_mse, expected_mse)

    # PSNR Test
    expected_psnr = 39.52742354296917
    computed_psnr = calculate_psnr(original, processed)

    print("\nPSNR Test")
    print("Expected PSNR:", expected_psnr)
    print("Computed PSNR:", computed_psnr)

    assert np.isclose(computed_psnr, expected_psnr)

    # Identical Images Test
    identical_psnr = calculate_psnr(original, original)

    print("\nIdentical Images PSNR:", identical_psnr)

    assert np.isinf(identical_psnr)

    # IoU Test
    mask_a = np.array([1, 1, 0, 0])
    mask_b = np.array([1, 0, 0, 0])

    expected_iou = 0.5
    computed_iou = calculate_iou(mask_a, mask_b)

    print("\nIoU Test")
    print("Expected IoU:", expected_iou)
    print("Computed IoU:", computed_iou)

    assert np.isclose(computed_iou, expected_iou)

    # Empty Masks Test
    empty_a = np.zeros((5, 5))
    empty_b = np.zeros((5, 5))

    empty_iou = calculate_iou(empty_a, empty_b)

    print("\nEmpty Masks IoU:", empty_iou)

    assert empty_iou == 1.0

    # Disjoint Masks Test
    disjoint_a = np.array([1, 1, 0, 0])
    disjoint_b = np.array([0, 0, 1, 1])

    disjoint_iou = calculate_iou(disjoint_a, disjoint_b)

    print("\nDisjoint Masks IoU:", disjoint_iou)

    assert disjoint_iou == 0.0

    print("\nAll Hand-Crafted Tests Passed!")
    print("=" * 50)


# ==========================================
# Cell 5: Kvasir-SEG Dataset Evaluation
# ==========================================

def evaluate_dataset_sample():

    print("\n" + "=" * 50)
    print("Kvasir-SEG Dataset Evaluation")
    print("=" * 50)

    pairs = get_image_mask_pairs()

    if len(pairs) == 0:
        raise ValueError("No image-mask pairs found")

    image_path, mask_path = pairs[0]

    image = load_image(image_path)
    ground_truth = load_ground_truth(mask_path)

    print("\nSample Name:", image_path.stem)
    print("Image Shape:", image.shape)
    print("Mask Shape:", ground_truth.shape)
    print("Polyp Pixels:", int(np.sum(ground_truth > 0)))

    # Image Quality Evaluation
    noisy_image = degrade_image(
        image,
        method="gaussian",
        seed=42
    )

    restored_image = cv2.medianBlur(
        noisy_image,
        3
    )

    noisy_mse = calculate_mse(image, noisy_image)
    noisy_psnr = calculate_psnr(image, noisy_image)

    restored_mse = calculate_mse(image, restored_image)
    restored_psnr = calculate_psnr(image, restored_image)

    print("\nImage Quality Evaluation")

    print(
        f"Noisy Image    | MSE: {noisy_mse:.2f} "
        f"| PSNR: {noisy_psnr:.2f} dB"
    )

    print(
        f"Restored Image | MSE: {restored_mse:.2f} "
        f"| PSNR: {restored_psnr:.2f} dB"
    )

    # IoU Demonstration
    kernel = np.ones((3, 3), dtype=np.uint8)

    dilated_mask = cv2.dilate(
        ground_truth.astype(np.uint8),
        kernel,
        iterations=1
    )

    eroded_mask = cv2.erode(
        ground_truth.astype(np.uint8),
        kernel,
        iterations=1
    )

    dilated_iou = calculate_iou(
        dilated_mask,
        ground_truth
    )

    eroded_iou = calculate_iou(
        eroded_mask,
        ground_truth
    )

    print("\nIoU Demonstration")
    print(f"Dilated Mask IoU: {dilated_iou:.4f}")
    print(f"Eroded Mask IoU: {eroded_iou:.4f}")

    print("\nDataset Evaluation Completed!")


# ==========================================
# Cell 6: Main
# ==========================================

def main():

    run_hand_crafted_tests()

    evaluate_dataset_sample()


if __name__ == "__main__":
    main()
