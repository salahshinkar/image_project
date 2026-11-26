
# ==========================================
# data.py
# Image Loading, Resizing, Splitting,
# Ground Truth, Controlled Degradation
# and Restoration Filters
# ==========================================


# ==========================================
# Cell 1: Import Libraries
# ==========================================

import zipfile
import cv2
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split


# ==========================================
# Cell 2: Dataset Paths
# ==========================================

# Project main folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset ZIP file
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.zip"

# Output folders
IMAGES_OUTPUT = PROJECT_ROOT / "data" / "resized_dataset" / "images"
MASKS_OUTPUT = PROJECT_ROOT / "data" / "resized_dataset" / "masks"


# ==========================================
# Cell 3: Read ZIP Dataset
# ==========================================

def open_dataset():

    # Open ZIP dataset
    dataset = zipfile.ZipFile(DATASET_PATH, "r")

    return dataset


# ==========================================
# Cell 4: Get Image and Mask Paths
# ==========================================

def get_image_mask_paths(dataset):

    # Get all files inside ZIP
    file_names = dataset.namelist()

    # Supported image extensions
    extensions = (".jpg", ".jpeg", ".png", ".bmp")

    # Store image paths
    image_paths = []

    # Store mask paths
    mask_paths = {}

    for file in file_names:

        # Check image extension
        if file.lower().endswith(extensions):

            # Check if file belongs to masks folder
            if "/masks/" in file.lower() or file.lower().startswith("masks/"):

                name = Path(file).stem
                mask_paths[name] = file

            # Check if file belongs to images folder
            elif "/images/" in file.lower() or file.lower().startswith("images/"):

                image_paths.append(file)

    return image_paths, mask_paths


# ==========================================
# Cell 5: Load Image from ZIP
# ==========================================

def load_image_from_zip(dataset, image_path):

    # Read image bytes
    image_bytes = dataset.read(image_path)

    # Convert bytes to NumPy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Check image
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# ==========================================
# Cell 6: Load Ground Truth Mask from ZIP
# ==========================================

def load_mask_from_zip(dataset, mask_path):

    # Read mask bytes
    mask_bytes = dataset.read(mask_path)

    # Convert bytes to NumPy array
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)

    # Decode mask as grayscale
    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

    # Check mask
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    # Convert mask to binary: 0 or 1
    mask = (mask > 0).astype(np.uint8)

    return mask


# ==========================================
# Cell 7: Resize Images and Masks
# ==========================================

def resize_dataset(dataset, image_paths, mask_paths, size=(512, 512)):

    # Create output folders
    IMAGES_OUTPUT.mkdir(parents=True, exist_ok=True)
    MASKS_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Count saved images and masks
    image_count = 0
    mask_count = 0

    for image_path in image_paths:

        # Get image name
        name = Path(image_path).stem

        # Load original image
        image = load_image_from_zip(dataset, image_path)

        # Resize image
        resized_image = cv2.resize(
            image,
            size,
            interpolation=cv2.INTER_AREA
        )

        # Convert RGB to BGR before saving
        resized_image = cv2.cvtColor(
            resized_image,
            cv2.COLOR_RGB2BGR
        )

        # Save resized image
        cv2.imwrite(
            str(IMAGES_OUTPUT / f"{name}.png"),
            resized_image
        )

        image_count += 1

        # Check if corresponding mask exists
        if name in mask_paths:

            # Load original mask
            mask = load_mask_from_zip(
                dataset,
                mask_paths[name]
            )

            # Resize mask using Nearest Neighbor
            resized_mask = cv2.resize(
                mask,
                size,
                interpolation=cv2.INTER_NEAREST
            )

            # Save resized mask
            cv2.imwrite(
                str(MASKS_OUTPUT / f"{name}.png"),
                resized_mask * 255
            )

            mask_count += 1

    print("Resizing Completed!")
    print("Images Saved:", image_count)
    print("Masks Saved:", mask_count)


# ==========================================
# Cell 8: Get Resized Image-Mask Pairs
# ==========================================

def get_image_mask_pairs():

    # Get image files
    image_files = sorted(IMAGES_OUTPUT.glob("*.png"))

    # Get mask files
    mask_files = sorted(MASKS_OUTPUT.glob("*.png"))

    # Create mask dictionary
    mask_dict = {
        mask.stem: mask
        for mask in mask_files
    }

    # Match images with masks
    pairs = []

    for image_path in image_files:

        name = image_path.stem

        if name in mask_dict:

            pairs.append(
                (image_path, mask_dict[name])
            )

    return pairs


# ==========================================
# Cell 9: Load Resized Image
# ==========================================

def load_image(image_path):

    # Read image
    image = cv2.imread(str(image_path))

    # Check image
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ==========================================
# Cell 10: Load Resized Ground Truth
# ==========================================

def load_ground_truth(mask_path):

    # Read mask as grayscale
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    # Check mask
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    # Convert mask to binary: 0 or 1
    mask = (mask > 0).astype(np.uint8)

    return mask


# ==========================================
# Cell 11: Split Dataset
# ==========================================

def split_dataset(pairs, test_size=0.30, random_state=42):

    # Get training and testing data
    train_pairs, test_pairs = train_test_split(
        pairs,
        test_size=test_size,
        random_state=random_state
    )

    print("Training Samples:", len(train_pairs))
    print("Testing Samples:", len(test_pairs))

    return train_pairs, test_pairs


# ==========================================
# Cell 12: Add Gaussian Noise
# ==========================================

def add_gaussian_noise(image, rng, mean=0, sigma=25):

    # Convert image to float
    image_float = image.astype(np.float32)

    # Generate Gaussian noise
    noise = rng.normal(
        loc=mean,
        scale=sigma,
        size=image.shape
    )

    # Add noise to image
    noisy_image = image_float + noise

    # Keep values between 0 and 255
    noisy_image = np.clip(
        noisy_image,
        0,
        255
    )

    # Convert back to uint8
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


# ==========================================
# Cell 13: Add Salt and Pepper Noise
# ==========================================

def add_salt_pepper_noise(image, rng, amount=0.05):

    # Copy original image
    noisy_image = image.copy()

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate number of noisy pixels
    num_pixels = int(amount * height * width)

    # Select random pixel positions
    positions = rng.choice(
        height * width,
        size=num_pixels,
        replace=False
    )

    # Flatten image pixels
    flat_image = noisy_image.reshape(-1, image.shape[2])

    # Half pixels become white (Salt)
    salt_pixels = positions[:num_pixels // 2]

    flat_image[salt_pixels] = 255

    # Remaining pixels become black (Pepper)
    pepper_pixels = positions[num_pixels // 2:]

    flat_image[pepper_pixels] = 0

    return noisy_image


# ==========================================
# Cell 14: Add Speckle Noise
# ==========================================

def add_speckle_noise(image, rng, sigma=0.1):

    # Convert image to float
    image_float = image.astype(np.float32)

    # Generate Gaussian noise
    noise = rng.normal(
        loc=0,
        scale=sigma,
        size=image.shape
    )

    # Apply multiplicative noise
    noisy_image = image_float + (image_float * noise)

    # Keep values between 0 and 255
    noisy_image = np.clip(
        noisy_image,
        0,
        255
    )

    # Convert back to uint8
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


# ==========================================
# Cell 15: Add Poisson Noise
# ==========================================

def add_poisson_noise(image, rng, levels=255):

    # Normalize image between 0 and 1
    image_float = image.astype(np.float32) / 255.0

    # Generate Poisson noise
    noisy_image = rng.poisson(
        image_float * levels
    ) / levels

    # Convert back to range 0 to 255
    noisy_image = noisy_image * 255

    # Keep values between 0 and 255
    noisy_image = np.clip(
        noisy_image,
        0,
        255
    )

    # Convert back to uint8
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


# ==========================================
# Cell 16: Apply Gaussian Blur
# ==========================================

def apply_gaussian_blur(image, kernel_size=5):

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

    return blurred_image


# ==========================================
# Cell 17: Controlled Degradation
# ==========================================

def degrade_image(image, method="gaussian", seed=42):

    # Create random generator with fixed seed
    rng = np.random.default_rng(seed)

    # Gaussian Noise
    if method == "gaussian":

        degraded_image = add_gaussian_noise(
            image,
            rng,
            mean=0,
            sigma=25
        )

    # Salt and Pepper Noise
    elif method == "salt_pepper":

        degraded_image = add_salt_pepper_noise(
            image,
            rng,
            amount=0.05
        )

    # Speckle Noise
    elif method == "speckle":

        degraded_image = add_speckle_noise(
            image,
            rng,
            sigma=0.1
        )

    # Poisson Noise
    elif method == "poisson":

        degraded_image = add_poisson_noise(
            image,
            rng,
            levels=255
        )

    # Gaussian Blur (Additional degradation)
    elif method == "blur":

        degraded_image = apply_gaussian_blur(
            image,
            kernel_size=5
        )

    else:

        raise ValueError(
            "Method must be: gaussian, salt_pepper, speckle, poisson, or blur"
        )

    return degraded_image


# ==========================================
# Cell 18: Mean Filter
# ==========================================

def apply_mean_filter(image, kernel_size=5):

    # Apply Mean Filter
    filtered_image = cv2.blur(
        image,
        (kernel_size, kernel_size)
    )

    return filtered_image


# ==========================================
# Cell 19: Gaussian Filter
# ==========================================

def apply_gaussian_filter(image, kernel_size=5):

    # Apply Gaussian Filter
    filtered_image = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

    return filtered_image


# ==========================================
# Cell 20: Median Filter
# ==========================================

def apply_median_filter(image, kernel_size=5):

    # Apply Median Filter
    filtered_image = cv2.medianBlur(
        image,
        kernel_size
    )

    return filtered_image


# ==========================================
# Cell 21: Main Function
# ==========================================

def main():

    # Open dataset
    dataset = open_dataset()

    try:

        # Get image and mask paths
        image_paths, mask_paths = get_image_mask_paths(dataset)

        print("Total Images:", len(image_paths))
        print("Total Masks:", len(mask_paths))

        # Resize images and masks
        resize_dataset(
            dataset,
            image_paths,
            mask_paths,
            size=(512, 512)
        )

        # Get image-mask pairs
        pairs = get_image_mask_pairs()

        print("Total Matched Pairs:", len(pairs))

        # Split dataset
        train_pairs, test_pairs = split_dataset(pairs)

        # Load one sample
        if len(train_pairs) > 0:

            image_path, mask_path = train_pairs[0]

            image = load_image(image_path)
            mask = load_ground_truth(mask_path)

            # Apply Gaussian Noise
            noisy_image = degrade_image(
                image,
                method="gaussian",
                seed=42
            )

            # Apply restoration filters
            mean_image = apply_mean_filter(noisy_image)

            gaussian_image = apply_gaussian_filter(noisy_image)

            median_image = apply_median_filter(noisy_image)

            # Print image information
            print("Image Shape:", image.shape)
            print("Mask Shape:", mask.shape)
            print("Noisy Image Shape:", noisy_image.shape)

    finally:

        # Close ZIP dataset
        dataset.close()


# ==========================================
# Cell 22: Run Main Function
# ==========================================

if __name__ == "__main__":

    main()







