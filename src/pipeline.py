
# ==========================================
# pipeline.py
# Baseline Segmentation and Restoration
# ==========================================

import cv2
import numpy as np

# ==========================================
# Imports
# ==========================================

if __package__:
    from .data import (
        get_image_mask_pairs,
        split_dataset,
        load_image,
        load_ground_truth,
        degrade_image
    )

    from .metrics import (
        calculate_mse,
        calculate_psnr,
        calculate_iou
    )

else:
    from pathlib import Path
    import sys

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
        calculate_mse,
        calculate_psnr,
        calculate_iou
    )


# ==========================================
# Cell 1: Prepare Dataset Split
# ==========================================

def prepare_dataset_split():

    pairs = get_image_mask_pairs()

    if len(pairs) == 0:
        raise ValueError("No image-mask pairs found")

    train_pairs, test_pairs = split_dataset(
        pairs,
        test_size=0.30,
        random_state=42
    )

    return train_pairs, test_pairs


# ==========================================
# Cell 2: Trivial Segmentation
# Global Threshold = 127
# ==========================================

def trivial_segmentation(image):

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image.copy()

    _, predicted_mask = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return (predicted_mask > 0).astype(np.uint8)


# ==========================================
# Cell 3: Simple Segmentation
# Otsu + 3x3 Opening
# ==========================================

def simple_segmentation(image):

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image.copy()

    _, predicted_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    opened_mask = cv2.morphologyEx(
        predicted_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return (opened_mask > 0).astype(np.uint8)


# ==========================================
# Cell 4: Trivial Restoration
# No Processing
# ==========================================

def trivial_restoration(degraded_image):

    return degraded_image.copy()


# ==========================================
# Cell 5: Simple Restoration
# 3x3 Median Filter
# ==========================================

def simple_restoration(degraded_image):

    restored_image = cv2.medianBlur(
        degraded_image,
        3
    )

    return restored_image


# ==========================================
# Cell 6: Evaluate Segmentation Baselines
# ==========================================

def evaluate_segmentation_baselines(test_pairs):

    trivial_ious = []
    simple_ious = []

    for image_path, mask_path in test_pairs:

        image = load_image(image_path)

        ground_truth = load_ground_truth(mask_path)

        trivial_pred = trivial_segmentation(image)

        simple_pred = simple_segmentation(image)

        trivial_iou = calculate_iou(
            trivial_pred,
            ground_truth
        )

        simple_iou = calculate_iou(
            simple_pred,
            ground_truth
        )

        trivial_ious.append(trivial_iou)

        simple_ious.append(simple_iou)

    results = {
        "trivial_iou": np.mean(trivial_ious),
        "simple_iou": np.mean(simple_ious)
    }

    return results


# ==========================================
# Cell 7: Evaluate Restoration Baselines
# ==========================================

def evaluate_restoration_baselines(
    test_pairs,
    noise_method="gaussian",
    seed=42
):

    trivial_mses = []
    trivial_psnrs = []

    simple_mses = []
    simple_psnrs = []

    for index, (image_path, mask_path) in enumerate(test_pairs):

        original_image = load_image(image_path)

        degraded_image = degrade_image(
            original_image,
            method=noise_method,
            seed=seed + index
        )

        trivial_result = trivial_restoration(
            degraded_image
        )

        simple_result = simple_restoration(
            degraded_image
        )

        # Trivial Restoration Metrics
        trivial_mse = calculate_mse(
            original_image,
            trivial_result
        )

        trivial_psnr = calculate_psnr(
            original_image,
            trivial_result
        )

        # Simple Restoration Metrics
        simple_mse = calculate_mse(
            original_image,
            simple_result
        )

        simple_psnr = calculate_psnr(
            original_image,
            simple_result
        )

        trivial_mses.append(trivial_mse)
        trivial_psnrs.append(trivial_psnr)

        simple_mses.append(simple_mse)
        simple_psnrs.append(simple_psnr)

    results = {
        "trivial_mse": np.mean(trivial_mses),
        "trivial_psnr": np.mean(trivial_psnrs),
        "simple_mse": np.mean(simple_mses),
        "simple_psnr": np.mean(simple_psnrs)
    }

    return results


# ==========================================
# Cell 8: Evaluate All Baselines
# ==========================================

def evaluate_baselines(test_pairs):

    segmentation_results = evaluate_segmentation_baselines(
        test_pairs
    )

    restoration_results = evaluate_restoration_baselines(
        test_pairs
    )

    results = {
        **segmentation_results,
        **restoration_results
    }

    return results


# ==========================================
# Cell 9: Run Baselines
# ==========================================

def run_baselines():

    train_pairs, test_pairs = prepare_dataset_split()

    print("=" * 50)
    print("DATASET SPLIT")
    print("=" * 50)

    print("Training Samples:", len(train_pairs))
    print("Testing Samples:", len(test_pairs))

    results = evaluate_baselines(test_pairs)

    print("\n" + "=" * 50)
    print("BASELINE RESULTS")
    print("=" * 50)

    print("\nSegmentation Results")

    print(
        f"Trivial Baseline IoU: "
        f"{results['trivial_iou']:.4f}"
    )

    print(
        f"Simple Baseline IoU: "
        f"{results['simple_iou']:.4f}"
    )

    print("\nRestoration Results")

    print(
        f"Trivial Baseline MSE: "
        f"{results['trivial_mse']:.2f}"
    )

    print(
        f"Trivial Baseline PSNR: "
        f"{results['trivial_psnr']:.2f} dB"
    )

    print(
        f"Simple Baseline MSE: "
        f"{results['simple_mse']:.2f}"
    )

    print(
        f"Simple Baseline PSNR: "
        f"{results['simple_psnr']:.2f} dB"
    )

    print("\nBaseline Evaluation Completed!")

    return train_pairs, test_pairs, results


# ==========================================
# Cell 10: Main
# ==========================================

if __name__ == "__main__":

    train_pairs, test_pairs, results = run_baselines()

