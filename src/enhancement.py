# ==========================================
# enhancement.py
# Image Enhancement Module:
# Histogram Analysis, Point Transformations,
# Histogram Equalization, CLAHE,
# Smoothing & Sharpening Filters,
# Kernel-Size Study, and Enhancement Comparison
# ==========================================


# ==========================================
# Cell 1: Import Libraries
# ==========================================

import os
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict


# ==========================================
# Cell 2: Dataset Path Configuration & Loader
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Search paths for dataset in priority order (matches data.py and metrics.py)
CANDIDATE_DATASET_PATHS = [
    PROJECT_ROOT / "data" / "dataset.zip",
    PROJECT_ROOT / "data" / "kvasir-seg.zip",
    Path(r"c:\Users\ohmar\OneDrive\Desktop\kvasir-seg.zip"),
    Path.home() / "Desktop" / "kvasir-seg.zip",
]


def get_dataset_path() -> Optional[Path]:
    """
    Locate the available dataset zip file.
    """
    for candidate in CANDIDATE_DATASET_PATHS:
        if candidate.is_file():
            return candidate
    return None


def load_dataset_sample(
    sample_index: int = 0,
    zip_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load RGB image and ground-truth binary mask from Kvasir-SEG dataset.
    Maintains exact original order from dataset.namelist().
    """
    if zip_path is None:
        zip_path = get_dataset_path()

    if zip_path is None or not Path(zip_path).is_file():
        raise FileNotFoundError(
            f"Dataset archive not found. Checked: {[str(p) for p in CANDIDATE_DATASET_PATHS]}"
        )

    zip_path = Path(zip_path)
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    with zipfile.ZipFile(zip_path, "r") as archive:
        all_files = archive.namelist()

        image_files = []
        mask_map = {}

        for file in all_files:
            if file.lower().endswith(image_extensions):
                if "/masks/" in file.lower() or file.lower().startswith("masks/"):
                    name = Path(file).stem
                    mask_map[name] = file
                elif "/images/" in file.lower() or file.lower().startswith("images/"):
                    image_files.append(file)

        if not image_files:
            raise ValueError(f"No image files found inside {zip_path}")

        sample_img_file = image_files[sample_index % len(image_files)]
        sample_name = Path(sample_img_file).stem

        if sample_name not in mask_map:
            raise ValueError(f"Matching mask for '{sample_name}' not found")

        sample_mask_file = mask_map[sample_name]

        # Decode image (RGB)
        img_bytes = archive.read(sample_img_file)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Decode mask (Binary uint8: 0 or 1)
        mask_bytes = archive.read(sample_mask_file)
        mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        mask_binary = (mask > 127).astype(np.uint8)

        return image, mask_binary, sample_name


# ==========================================
# Cell 3: Histogram Analysis
# ==========================================

def compute_histogram(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute 256-bin intensity histograms for grayscale or RGB channels.
    """
    histograms = {}

    if image.ndim == 2:
        hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
        histograms["gray"] = hist.flatten()
    elif image.ndim == 3:
        colors = ("red", "green", "blue")
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
            histograms[color] = hist.flatten()

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist_gray = cv2.calcHist([gray], [0], mask, [256], [0, 256])
        histograms["intensity"] = hist_gray.flatten()

    return histograms


def analyze_histogram_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Compute fundamental statistical properties of the image intensity distribution.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    dynamic_range = max_val - min_val

    # Intensity entropy (measure of information content)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy_val = float(-np.sum(hist * np.log2(hist)))

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "dynamic_range": dynamic_range,
        "entropy": entropy_val
    }


# ==========================================
# Cell 4: Point Transformations
# ==========================================

def log_transform(
    image: np.ndarray,
    c: Optional[float] = None
) -> np.ndarray:
    """
    Logarithmic transformation: s = c * log(1 + r).
    Expands the dark regions while compressing the high bright values.
    """
    img_float = image.astype(np.float64)
    if c is None:
        c = 255.0 / np.log(1.0 + np.max(img_float))

    transformed = c * np.log(1.0 + img_float)
    return np.clip(transformed, 0, 255).astype(np.uint8)


def power_law_transform(
    image: np.ndarray,
    gamma: float = 1.0,
    c: float = 1.0
) -> np.ndarray:
    """
    Power-Law (Gamma) transformation: s = c * (r / 255)^gamma * 255.
    - gamma < 1: brightens dark regions (useful for colon shadows)
    - gamma > 1: darkens mid-tones and enhances bright contrast
    """
    normalized = image.astype(np.float64) / 255.0
    transformed = c * (normalized ** gamma) * 255.0
    return np.clip(transformed, 0, 255).astype(np.uint8)


def contrast_stretching(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    """
    Linear contrast stretching (normalization) using percentile clipping
    to eliminate specular reflection outliers.
    """
    img_float = image.astype(np.float64)

    if image.ndim == 3:
        stretched = np.zeros_like(img_float)
        for ch in range(3):
            r_min = np.percentile(img_float[..., ch], lower_percentile)
            r_max = np.percentile(img_float[..., ch], upper_percentile)
            if r_max > r_min:
                stretched[..., ch] = (img_float[..., ch] - r_min) * (255.0 / (r_max - r_min))
            else:
                stretched[..., ch] = img_float[..., ch]
    else:
        r_min = np.percentile(img_float, lower_percentile)
        r_max = np.percentile(img_float, upper_percentile)
        if r_max > r_min:
            stretched = (img_float - r_min) * (255.0 / (r_max - r_min))
        else:
            stretched = img_float

    return np.clip(stretched, 0, 255).astype(np.uint8)


def image_negative(image: np.ndarray) -> np.ndarray:
    """
    Negative transformation: s = 255 - r.
    Inverts intensities; highlights subtle dark vascular patterns in mucosa.
    """
    return 255 - image


# ==========================================
# Cell 5: Histogram Equalization & CLAHE
# ==========================================

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Global Histogram Equalization.
    For RGB images, applies equalization to the Luminance (Y) channel in YCrCb color space
    to enhance global contrast without corrupting polyp chromaticity.
    """
    if image.ndim == 2:
        return cv2.equalizeHist(image)

    # Convert RGB to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Limits noise amplification and glare in mucosal specular reflections.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        return clahe.apply(image)

    # Apply CLAHE on L channel in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ==========================================
# Cell 6: Spatial Smoothing Filters
# ==========================================

def apply_box_filter(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Linear spatial averaging (Box filter).
    """
    return cv2.blur(image, (kernel_size, kernel_size))


def apply_gaussian_smoothing(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.5
) -> np.ndarray:
    """
    Gaussian smoothing filter with standard deviation sigma.
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_bilateral_smoothing(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Edge-preserving Bilateral Filter.
    Smooths mucosal background textures while keeping sharp polyp boundaries intact.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ==========================================
# Cell 7: Spatial Sharpening Filters
# ==========================================

def apply_laplacian_sharpening(
    image: np.ndarray,
    c: float = 1.0
) -> np.ndarray:
    """
    Laplacian sharpening filter:
    g(x, y) = f(x, y) + c * [-nabla^2 f(x, y)]
    """
    # Standard 3x3 Laplacian kernel with negative center
    laplacian_kernel = np.array([
        [0, -1,  0],
        [-1,  4, -1],
        [0, -1,  0]
    ], dtype=np.float64)

    img_float = image.astype(np.float64)

    if image.ndim == 3:
        sharpened = np.zeros_like(img_float)
        for ch in range(3):
            lap = cv2.filter2D(img_float[..., ch], -1, laplacian_kernel)
            sharpened[..., ch] = img_float[..., ch] + c * lap
    else:
        lap = cv2.filter2D(img_float, -1, laplacian_kernel)
        sharpened = img_float + c * lap

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_unsharp_mask(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.5,
    amount: float = 1.5,
    threshold: int = 0
) -> np.ndarray:
    """
    Unsharp Masking / High-boost filtering:
    mask = image - blurred
    enhanced = image + amount * mask
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    diff = image.astype(np.float64) - blurred.astype(np.float64)

    if threshold > 0:
        low_contrast_mask = np.abs(diff) < threshold
        diff[low_contrast_mask] = 0.0

    sharpened = image.astype(np.float64) + amount * diff
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ==========================================
# Cell 8: Kernel-Size Study (Required by Proposal)
# ==========================================

def run_kernel_size_study(
    image: np.ndarray,
    kernel_sizes: Optional[List[int]] = None
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Investigate the effect of kernel sizes on smoothing and sharpening filters.
    Sizes: 3, 5, 7, 9, 11
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 5, 7, 9, 11]

    results = {
        "gaussian": {},
        "mean": {},
        "unsharp": {}
    }

    for k in kernel_sizes:
        # Gaussian smoothing
        results["gaussian"][k] = apply_gaussian_smoothing(image, kernel_size=k, sigma=k / 3.0)
        # Box/Mean smoothing
        results["mean"][k] = apply_box_filter(image, kernel_size=k)
        # Unsharp masking
        results["unsharp"][k] = apply_unsharp_mask(image, kernel_size=k, sigma=k / 3.0, amount=1.2)

    return results


# ==========================================
# Cell 9: Enhancement Comparison Study (Required by Proposal)
# ==========================================

def run_enhancement_comparison(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate all primary enhancement candidate methods for comparative evaluation:
    1. Original Image (Baseline)
    2. Log Transformation
    3. Power-Law / Gamma Transformation (gamma = 0.6)
    4. Contrast Stretching (1% - 99% Percentiles)
    5. Global Histogram Equalization (YCrCb space)
    6. CLAHE (LAB space, clipLimit = 2.0)
    7. Edge-Preserving Bilateral Smoothing
    8. Unsharp Masking Sharpening
    """
    return {
        "Original": image,
        "Log Transform": log_transform(image),
        "Gamma (0.6)": power_law_transform(image, gamma=0.6),
        "Contrast Stretch": contrast_stretching(image),
        "Global HE": histogram_equalization(image),
        "CLAHE": apply_clahe(image, clip_limit=2.0),
        "Bilateral Filter": apply_bilateral_smoothing(image),
        "Unsharp Masking": apply_unsharp_mask(image, kernel_size=5, amount=1.5)
    }


# ==========================================
# Cell 10: Main Function & Verification
# ==========================================

def main() -> None:
    """
    Demonstrate all enhancement functions on the primary Kvasir-SEG dataset sample.
    """
    print("=" * 60)
    print("Running Image Enhancement Module on Kvasir-SEG Dataset")
    print("=" * 60)

    # 1. Load dataset sample
    image, mask, sample_name = load_dataset_sample(sample_index=0)
    print(f"Loaded Sample       : {sample_name}.jpg")
    print(f"Original Image Shape: {image.shape}")
    print(f"Original Mask Shape : {mask.shape} (Polyp pixels: {mask.sum()})")

    # 2. Histogram Statistics
    stats = analyze_histogram_stats(image)
    print("\n[Histogram Statistics - Original Image]")
    print(f"  Mean Intensity : {stats['mean']:.2f}")
    print(f"  Std Deviation  : {stats['std']:.2f}")
    print(f"  Dynamic Range  : [{stats['min']:.0f}, {stats['max']:.0f}] (Span: {stats['dynamic_range']:.0f})")
    print(f"  Entropy        : {stats['entropy']:.2f} bits")

    # 3. Enhancement Comparison Study
    print("\n[Running Enhancement Comparison Study]")
    enhanced_dict = run_enhancement_comparison(image)
    for name, enhanced_img in enhanced_dict.items():
        enh_stats = analyze_histogram_stats(enhanced_img)
        print(f"  {name:<18} -> Mean: {enh_stats['mean']:6.2f} | Std: {enh_stats['std']:5.2f} | Entropy: {enh_stats['entropy']:4.2f}")

    # 4. Kernel-Size Study
    print("\n[Running Kernel-Size Study (Sizes: 3, 5, 7, 9, 11)]")
    kernel_results = run_kernel_size_study(image, [3, 5, 7, 9, 11])
    for k in [3, 5, 7, 9, 11]:
        gauss_std = np.std(kernel_results["gaussian"][k])
        unsharp_std = np.std(kernel_results["unsharp"][k])
        print(f"  Kernel {k:2d}x{k:<2d} -> Gaussian Std: {gauss_std:5.2f} | Unsharp Mask Std: {unsharp_std:5.2f}")

    print("\n" + "=" * 60)
    print("Image Enhancement Module Verification Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
