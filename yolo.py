import cv2
import numpy as np
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray
from skimage import img_as_float
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

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


def adaptive_gamma_correction(img, gamma=None):
    img_float = img_as_float(img)
    if gamma is None:
        avg = np.mean(img_float)
        gamma = 1.0 if 0.4 < avg < 0.6 else (0.8 if avg > 0.6 else 1.2)
    corrected = np.power(img_float, gamma)
    return np.clip(corrected, 0, 1)


def contrast_improvement_index(original, enhanced):
    std_original = np.std(original)
    return np.std(enhanced) / std_original if std_original != 0 else float('inf')


def enhance_foggy_image(image_path, show_result=True, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid format.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Select ROI and press ENTER or SPACE. Press ESC to cancel.")
    roi = cv2.selectROI("Select ROI", img_rgb)
    cv2.destroyWindow("Select ROI")
    if sum(roi) == 0:
        print("No ROI selected, using full image.")
    else:
        x, y, w, h = roi
        img_rgb = img_rgb[y:y+h, x:x+w]

    img_float = img_as_float(img_rgb)

    dark = dark_channel(img_float)
    A = estimate_atmospheric_light(img_float, dark)
    transmission = estimate_transmission(img_float, A)
    dcp_result = recover_image(img_float, A, transmission)

    dcp_result_uint8 = (dcp_result * 255).astype(np.uint8)
    clahe_img = apply_clahe_color(dcp_result_uint8)

    median_filtered = cv2.medianBlur(clahe_img, 3)
    bilateral_filtered = cv2.bilateralFilter(median_filtered, d=9, sigmaColor=75, sigmaSpace=75)

    agc_img = adaptive_gamma_correction(bilateral_filtered)
    final_result = (agc_img * 255).astype(np.uint8)

    gray_original = rgb2gray(img_rgb)
    entropy_original = shannon_entropy(img_rgb)
    cii_original = contrast_improvement_index(gray_original, gray_original)

    gray_enhanced = rgb2gray(final_result)
    entropy_enhanced = shannon_entropy(final_result)
    cii_enhanced = contrast_improvement_index(gray_original, gray_enhanced)

    print("\n=== Metrics for Original Image ===")
    print(f"Shannon Entropy (Original): {entropy_original:.4f}")
    print(f"Contrast Improvement Index (Original): {cii_original:.4f}")

    print("\n=== Metrics for Enhanced Image ===")
    print(f"Shannon Entropy (Enhanced): {entropy_enhanced:.4f}")
    print(f"Contrast Improvement Index (Enhanced): {cii_enhanced:.4f}")

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))

    if show_result:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img_rgb)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Enhanced")
        plt.imshow(final_result)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return final_result



def detect_vehicles(enhanced_img):
    print("\n🔍 Loading YOLOv8 model for vehicle detection...")
    model = YOLO("yolov8n.pt") 

    print("Running detection...")
    results = model.predict(source=enhanced_img, imgsz=640, conf=0.40, verbose=False)

    annotated = results[0].plot()  
    cv2.imshow("Vehicle Detection Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    vehicle_classes = ['car', 'bus', 'truck', 'motorbike']
    count = sum(1 for r in results[0].boxes.cls if model.names[int(r)] in vehicle_classes)
    print(f"🚗 Total Vehicles Detected: {count}")

    return annotated



if __name__ == "__main__":
    input_path = "foggy4.jpg"        
    enhanced_path = "Enhanced.jpg"  
    detected_path = "Detected.jpg"  

    enhanced_img = enhance_foggy_image(
        image_path=input_path,
        show_result=True,
        save_path=enhanced_path
    )

    detection_result = detect_vehicles(enhanced_img)

    cv2.imwrite(detected_path, cv2.cvtColor(detection_result, cv2.COLOR_RGB2BGR))
    print(f"\n✅ Results saved as '{enhanced_path}' and '{detected_path}'")

