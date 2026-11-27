
# ==========================================
# pipeline.py
# Baseline Segmentation and Restoration
# ==========================================

# ==========================================
# Cell 1: Import Libraries
# ==========================================

import cv2
import numpy as np

# Support both direct execution and package execution
if __package__:
    from .data import (
        get_image_mask_pairs,
        split_dataset,
        load_image,
        load_ground_truth,
        degrade_image
    )

    from .metrics import (
        calculate_iou,
        calculate_mse,
        calculate_psnr
    )

else:
    import sys
    from pathlib import Path

    # Add project root to Python path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.data import (
        get_image_mask_pairs,
        split_dataset,
        load_image,
        load_ground_truth,
        degrade_image
    )

    from src.metrics import (
        calculate_iou,
        calculate_mse,
        calculate_psnr
    )


# ==========================================
# Cell 2: Prepare Train and Test Data
# ==========================================

def prepare_dataset_split():

    # Get all matched image-mask pairs
    pairs = get_image_mask_pairs()

    # Check dataset
    if len(pairs) == 0:
        raise ValueError(
            "No image-mask pairs found. Run resize_dataset() first."
        )

    # Split dataset into training and testing
    train_pairs, test_pairs = split_dataset(
        pairs,
        test_size=0.30,
        random_state=42
    )

    return train_pairs, test_pairs


# ==========================================
# Cell 3: Trivial Segmentation Baseline
# Fixed Global Threshold = 127
# ==========================================

def trivial_segmentation(image):

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Apply fixed threshold = 127
    _, predicted_mask = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Convert mask to binary: 0 or 1
    predicted_mask = (predicted_mask > 0).astype(np.uint8)

    return predicted_mask


# ==========================================
# Cell 4: Simple Segmentation Baseline
# Otsu Threshold + 3x3 Opening
# ==========================================

def simple_segmentation(image):

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Apply Otsu Thresholding
    _, predicted_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Create 3x3 morphological kernel
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    # Apply morphological opening once
    predicted_mask = cv2.morphologyEx(
        predicted_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Convert mask to binary: 0 or 1
    predicted_mask = (predicted_mask > 0).astype(np.uint8)

    return predicted_mask


# ==========================================
# Cell 5: Trivial Restoration Baseline
# Unprocessed Degraded Image
# ==========================================

def trivial_restoration(degraded_image):

    # Return the degraded image without processing
    return degraded_image.copy()


# ==========================================
# Cell 6: Simple Restoration Baseline
# 3x3 Median Filter
# ==========================================

def simple_restoration(degraded_image):

    # Apply 3x3 Median Filter
    restored_image = cv2.medianBlur(
        degraded_image,
        3
    )

    return restored_image


# ==========================================
# Cell 7: Evaluate Segmentation Baselines
# ==========================================

def evaluate_segmentation_baselines(test_pairs):

    # Store IoU results
    trivial_ious = []
    simple_ious = []

    # Evaluate only on test data
    for image_path, mask_path in test_pairs:

        # Load image and ground truth
        image = load_image(image_path)
        gt_mask = load_ground_truth(mask_path)

        # Generate predictions
        trivial_pred = trivial_segmentation(image)
        simple_pred = simple_segmentation(image)

        # Calculate IoU
        trivial_iou = calculate_iou(
            gt_mask,
            trivial_pred
        )

        simple_iou = calculate_iou(
            gt_mask,
            simple_pred
        )

        # Store results
        trivial_ious.append(trivial_iou)
        simple_ious.append(simple_iou)

    # Calculate average IoU
    results = {
        "trivial_iou": float(np.mean(trivial_ious)),
        "simple_iou": float(np.mean(simple_ious))
    }

    return results


# ==========================================
# Cell 8: Evaluate Restoration Baselines
# ==========================================

def evaluate_restoration_baselines(
    test_pairs,
    noise_method="gaussian",
    seed=42
):

    # Store evaluation results
    trivial_mse = []
    trivial_psnr = []

    simple_mse = []
    simple_psnr = []

    # Evaluate only on test data
    for index, (image_path, mask_path) in enumerate(test_pairs):

        # Load clean image
        clean_image = load_image(image_path)

        # Generate controlled degradation
        degraded_image = degrade_image(
            clean_image,
            method=noise_method,
            seed=seed + index
        )

        # Trivial restoration
        trivial_result = trivial_restoration(
            degraded_image
        )

        # Simple restoration
        simple_result = simple_restoration(
            degraded_image
        )

        # Calculate trivial metrics
        mse_trivial = calculate_mse(
            clean_image,
            trivial_result
        )

        psnr_trivial = calculate_psnr(
            clean_image,
            trivial_result
        )

        # Calculate simple metrics
        mse_simple = calculate_mse(
            clean_image,
            simple_result
        )

        psnr_simple = calculate_psnr(
            clean_image,
            simple_result
        )

        # Store results
        trivial_mse.append(mse_trivial)
        trivial_psnr.append(psnr_trivial)

        simple_mse.append(mse_simple)
        simple_psnr.append(psnr_simple)

    # Calculate average metrics
    results = {
        "trivial_mse": float(np.mean(trivial_mse)),
        "trivial_psnr": float(np.mean(trivial_psnr)),
        "simple_mse": float(np.mean(simple_mse)),
        "simple_psnr": float(np.mean(simple_psnr))
    }

    return results


# ==========================================
# Cell 9: Evaluate All Baselines
# ==========================================

def evaluate_baselines(test_pairs):

    # Evaluate segmentation
    segmentation_results = evaluate_segmentation_baselines(
        test_pairs
    )

    # Evaluate restoration
    restoration_results = evaluate_restoration_baselines(
        test_pairs,
        noise_method="gaussian",
        seed=42
    )

    # Combine results
    results = {
        **segmentation_results,
        **restoration_results
    }

    return results


# ==========================================
# Cell 10: Run Baseline Evaluation
# ==========================================

def run_baselines():

    # Split dataset
    train_pairs, test_pairs = prepare_dataset_split()

    print("=" * 50)
    print("DATASET SPLIT")
    print("=" * 50)

    print("Training Samples:", len(train_pairs))
    print("Testing Samples:", len(test_pairs))

    # Evaluate baselines
    results = evaluate_baselines(test_pairs)

    print("\n" + "=" * 50)
    print("BASELINE RESULTS")
    print("=" * 50)

    print("\nSegmentation Results")

    print(
        "Trivial Baseline IoU:",
        round(results["trivial_iou"], 4)
    )

    print(
        "Simple Baseline IoU:",
        round(results["simple_iou"], 4)
    )

    print("\nRestoration Results")

    print(
        "Trivial Baseline MSE:",
        round(results["trivial_mse"], 2)
    )

    print(
        "Trivial Baseline PSNR:",
        round(results["trivial_psnr"], 2),
        "dB"
    )

    print(
        "Simple Baseline MSE:",
        round(results["simple_mse"], 2)
    )

    print(
        "Simple Baseline PSNR:",
        round(results["simple_psnr"], 2),
        "dB"
    )

    print("\nBaseline Evaluation Completed!")

    return train_pairs, test_pairs, results


# ==========================================
# Cell 11: Main Execution
# ==========================================

if __name__ == "__main__":

    train_pairs, test_pairs, results = run_baselines()
