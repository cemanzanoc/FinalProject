import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove

def remove_background(image_path, output_folder="processed_images"):
    """
    Removes the background of the image using the rembg library.
    
    Parameters:
      - image_path (str): Path to the input image.
      - output_folder (str): Folder where the processed image will be saved.
      
    Returns:
      - processed_image_path (str): Path to the background-removed image.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Open the image and convert to RGBA to allow transparency
    image = Image.open(image_path).convert("RGBA")
    output = remove(image)
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    processed_image_path = os.path.join(output_folder, f"{image_name}_no_bg.png")
    output.save(processed_image_path)
    return processed_image_path


def remove_faces(image_path, output_folder="processed_images"):
    """
    Detects and removes (makes transparent) faces in an image.
    It is assumed that the image already has the background removed.
    
    Parameters:
      - image_path (str): Path to the image to process.
      - output_folder (str): Folder where the image with faces removed will be saved.
      
    Returns:
      - processed_face_path (str): Path to the image with faces removed.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the image with the alpha channel
    image_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # If the image does not have an alpha channel, add one
    if image_cv.shape[2] == 3:
        b, g, r = cv2.split(image_cv)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        image_cv = cv2.merge((b, g, r, alpha))
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_cv[:, :, :3], cv2.COLOR_BGR2GRAY)
    
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Make detected face areas transparent by setting alpha to 0
    for (x, y, w, h) in faces:
        image_cv[y:y+h, x:x+w, 3] = 0
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    processed_face_path = os.path.join(output_folder, f"{image_name}_no_faces.png")
    cv2.imwrite(processed_face_path, image_cv)
    return processed_face_path


def extract_dominant_colors(image_path, num_colors=3, min_percentage=1):
    """
    Extracts the dominant colors from an image using histogram analysis
    in the HSV color space.
    
    Parameters:
      - image_path (str): Path to the image.
      - num_colors (int): Number of main colors to detect.
      - min_percentage (float): Minimum percentage to consider a color.
      
    Returns:
      - A list of dominant color labels, e.g., ['red', 'blue', 'green'].
    """
    print("Analyzing dominant colors in the image...")
    
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram of the Hue channel
    hist = cv2.calcHist([image_hsv], [0], None, [180], [0, 180]).flatten()
    
    # Convert histogram values to percentages
    hist_percentage = (hist / hist.sum()) * 100
    
    # Sort indices by frequency (from highest to lowest)
    sorted_indices = np.argsort(hist_percentage)[::-1]
    top_colors = [hue for hue in sorted_indices if hist_percentage[hue] >= min_percentage][:num_colors]
    
    # Map Hue values to color labels
    color_labels = []
    for hue in top_colors:
        if 0 <= hue < 10 or hue >= 170:
            color_labels.append("red")
        elif 10 <= hue < 25:
            color_labels.append("orange")
        elif 25 <= hue < 35:
            color_labels.append("yellow")
        elif 35 <= hue < 85:
            color_labels.append("green")
        elif 85 <= hue < 130:
            color_labels.append("blue")
        elif 130 <= hue < 160:
            color_labels.append("purple")
        elif 160 <= hue < 170:
            color_labels.append("pink")
        elif 170 <= hue < 195:
            color_labels.append("black")
        elif 195 <= hue < 225:
            color_labels.append("white")
        elif 225 <= hue < 245:
            color_labels.append("beige")
        elif 245 <= hue < 265:
            color_labels.append("brown")
    
    # Remove duplicates and return
    return list(set(color_labels))
