import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray
from skimage import img_as_float

def dark_channel(img, window_size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def estimate_atmospheric_light(img, dark_channel):
    h, w = dark_channel.shape
    n_pixels = h * w
    n_brightest = int(max(n_pixels * 0.001, 1))
    flat_dark = dark_channel.ravel()
    flat_img = img.reshape(n_pixels, 3)
    indices = flat_dark.argsort()[-n_brightest:]
    brightest = flat_img[indices]
    A = np.max(brightest, axis=0)
    return A

def estimate_transmission(img, A, omega=0.95, window_size=15):
    normed = img / A
    transmission = 1 - omega * dark_channel(normed, window_size)
    return np.clip(transmission, 0.1, 1)

def recover_image(img, A, transmission):
    recovered = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        recovered[..., c] = (img[..., c] - A[c]) / transmission + A[c]
    return np.clip(recovered, 0, 1)

def apply_clahe_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def adaptive_gamma_correction(img):
    img_float = img_as_float(img)
    avg = np.mean(img_float)
    gamma = 0.5 if avg > 0.5 else 1.5
    corrected = np.power(img_float, gamma)
    return np.clip(corrected, 0, 1)

def contrast_improvement_index(original, enhanced):
    std_original = np.std(original)
    return np.std(enhanced) / std_original if std_original != 0 else float('inf')

foggy_dir = "fog_dataset/fog images"
output_dir = "fog_dataset/enhanced_results"
os.makedirs(output_dir, exist_ok=True)

entropy_list = []
cii_list = []

for filename in sorted(os.listdir(foggy_dir)):
    img_path = os.path.join(foggy_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)

    dark = dark_channel(img_float)
    A = estimate_atmospheric_light(img_float, dark)
    transmission = estimate_transmission(img_float, A)
    dcp_result = recover_image(img_float, A, transmission)

    dcp_uint8 = (dcp_result * 255).astype(np.uint8)
    clahe_img = apply_clahe_color(dcp_uint8)

    median_filtered = cv2.medianBlur(clahe_img, 3)
    bilateral_filtered = cv2.bilateralFilter(median_filtered, d=9, sigmaColor=75, sigmaSpace=75)

    final_img = adaptive_gamma_correction(bilateral_filtered)
    final_uint8 = (final_img * 255).astype(np.uint8)

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(final_uint8, cv2.COLOR_RGB2BGR))

    gray_original = rgb2gray(img_rgb)
    gray_enhanced = rgb2gray(final_uint8)
    entropy = shannon_entropy(final_uint8)
    cii = contrast_improvement_index(gray_original, gray_enhanced)

    entropy_list.append(entropy)
    cii_list.append(cii)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Enhanced")
    plt.imshow(final_uint8)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(entropy_list, marker='o')
plt.title("Shannon Entropy")
plt.xlabel("Image Index")
plt.ylabel("Entropy")

plt.subplot(1, 2, 2)
plt.plot(cii_list, marker='s', color='green')
plt.title("Contrast Improvement Index (CII)")
plt.xlabel("Image Index")
plt.ylabel("CII")

plt.tight_layout()
plt.show()
