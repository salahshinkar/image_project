
# ==========================================
# metrics.py
# Image Quality and Segmentation Evaluation
# MSE, PSNR, IoU
# Load Resized Dataset from data.py
# ==========================================

# ==========================================
# Cell 1: Import Libraries
# ==========================================

import cv2
import numpy as np

# ==========================================
# Import Data Functions
# ==========================================

if __package__:

    from .data import (
        get_image_mask_pairs,
        load_image,
        load_ground_truth
    )

else:

    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.data import (
        get_image_mask_pairs,
        load_image,
        load_ground_truth
    )


# ==========================================
# Cell 2: Mean Squared Error (MSE)
# ==========================================

def calculate_mse(y_true, y_pred):

    # Convert inputs to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")

    # Convert to float to avoid integer overflow
    diff = y_true.astype(np.float64) - y_pred.astype(np.float64)

    # Calculate MSE
    mse_value = np.mean(diff ** 2)

    return float(mse_value)


# ==========================================
# Cell 3: Peak Signal-to-Noise Ratio (PSNR)
# ==========================================

def calculate_psnr(y_true, y_pred, max_val=255.0):

    # Calculate MSE
    mse_value = calculate_mse(y_true, y_pred)

    # Identical images
    if mse_value == 0:
        return float("inf")

    # Calculate PSNR
    psnr_value = 10 * np.log10((max_val ** 2) / mse_value)

    return float(psnr_value)


# ==========================================
# Cell 4: Intersection over Union (IoU)
# ==========================================

def calculate_iou(y_true, y_pred, threshold=0.5):

    # Convert inputs to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError("Masks must have the same shape")

    # Convert masks to binary
    mask_true = y_true if y_true.dtype == bool else (y_true > threshold)
    mask_pred = y_pred if y_pred.dtype == bool else (y_pred > threshold)

    # Calculate intersection and union
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()

    # Handle empty masks
    if union == 0:
        return 1.0

    # Calculate IoU
    return float(intersection / union)


# ==========================================
# Cell 5: Load Resized Kvasir-SEG Sample
# ==========================================

def load_kvasir_sample(sample_index=0):

    # Get image-mask pairs from resized dataset
    pairs = get_image_mask_pairs()

    # Check dataset
    if len(pairs) == 0:
        raise ValueError(
            "No resized image-mask pairs found. "
            "Run resize_dataset() in data.py first."
        )

    # Check sample index
    if sample_index < 0 or sample_index >= len(pairs):
        raise IndexError(
            f"Sample index must be between 0 and {len(pairs) - 1}"
        )

    # Get selected image and mask paths
    image_path, mask_path = pairs[sample_index]

    # Load resized image using data.py
    image = load_image(image_path)

    # Load resized ground truth mask using data.py
    mask = load_ground_truth(mask_path)

    # Get sample name
    sample_name = image_path.stem

    return image, mask, sample_name


# ==========================================
# Cell 6: Hand-Crafted Verification Tests
# ==========================================

def run_hand_crafted_tests():

    print("=" * 50)
    print("Running Hand-Crafted Tests")
    print("=" * 50)

    # ------------------------------------------
    # Test 1: MSE
    # ------------------------------------------

    a_true = np.array(
        [[10, 20],
         [30, 40]],
        dtype=np.uint8
    )

    a_pred = np.array(
        [[12, 20],
         [25, 40]],
        dtype=np.uint8
    )

    mse = calculate_mse(a_true, a_pred)

    print("\nMSE Test")
    print("Expected MSE:", 7.25)
    print("Computed MSE:", mse)

    assert np.isclose(mse, 7.25)

    # ------------------------------------------
    # Test 2: PSNR
    # ------------------------------------------

    psnr = calculate_psnr(a_true, a_pred)

    expected_psnr = 10 * np.log10((255 ** 2) / 7.25)

    print("\nPSNR Test")
    print("Expected PSNR:", expected_psnr)
    print("Computed PSNR:", psnr)

    assert np.isclose(psnr, expected_psnr)

    # Identical images
    identical_psnr = calculate_psnr(a_true, a_true)

    print("\nIdentical Images PSNR:", identical_psnr)

    assert identical_psnr == float("inf")

    # ------------------------------------------
    # Test 3: IoU
    # ------------------------------------------

    mask_true = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.uint8)

    mask_pred = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 0]
    ], dtype=np.uint8)

    iou = calculate_iou(mask_true, mask_pred)

    print("\nIoU Test")
    print("Expected IoU:", 0.50)
    print("Computed IoU:", iou)

    assert np.isclose(iou, 0.50)

    # ------------------------------------------
    # Test 4: Empty Masks
    # ------------------------------------------

    empty_mask = np.zeros((3, 3), dtype=np.uint8)

    empty_iou = calculate_iou(empty_mask, empty_mask)

    print("\nEmpty Masks IoU:", empty_iou)

    assert empty_iou == 1.0

    # ------------------------------------------
    # Test 5: Disjoint Masks
    # ------------------------------------------

    mask_a = np.array([
        [1, 0],
        [0, 0]
    ])

    mask_b = np.array([
        [0, 1],
        [0, 0]
    ])

    disjoint_iou = calculate_iou(mask_a, mask_b)

    print("\nDisjoint Masks IoU:", disjoint_iou)

    assert disjoint_iou == 0.0

    print("\nAll Hand-Crafted Tests Passed!")


# ==========================================
# Cell 7: Kvasir-SEG Dataset Evaluation
# ==========================================

def run_kvasir_dataset_evaluation(sample_index=0):

    print("=" * 50)
    print("Kvasir-SEG Dataset Evaluation")
    print("=" * 50)

    # Load resized image and ground truth
    image, gt_mask, sample_name = load_kvasir_sample(sample_index)

    print("\nSample Name:", sample_name)
    print("Image Shape:", image.shape)
    print("Mask Shape:", gt_mask.shape)
    print("Polyp Pixels:", gt_mask.sum())

    # ------------------------------------------
    # Image Quality Evaluation
    # ------------------------------------------

    # Create controlled Gaussian noise
    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0,
        scale=20,
        size=image.shape
    )

    noisy_image = np.clip(
        image.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)

    # Apply Gaussian restoration filter
    restored_image = cv2.GaussianBlur(
        noisy_image,
        (5, 5),
        0
    )

    # Calculate metrics
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

    # ------------------------------------------
    # Segmentation Metric Demonstration
    # ------------------------------------------

    # Create a sample predicted mask using morphology
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    pred_dilated = cv2.dilate(
        gt_mask,
        kernel,
        iterations=1
    )

    pred_eroded = cv2.erode(
        gt_mask,
        kernel,
        iterations=1
    )

    # Calculate IoU
    iou_dilated = calculate_iou(gt_mask, pred_dilated)
    iou_eroded = calculate_iou(gt_mask, pred_eroded)

    print("\nIoU Demonstration")

    print("Dilated Mask IoU:", round(iou_dilated, 4))
    print("Eroded Mask IoU:", round(iou_eroded, 4))

    print("\nDataset Evaluation Completed!")


# ==========================================
# Cell 8: Main Execution
# ==========================================

if __name__ == "__main__":

    # Run mathematical tests
    run_hand_crafted_tests()

    # Run dataset evaluation
    run_kvasir_dataset_evaluation(sample_index=0)
