import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the CSV file containing the reference colors
# The CSV is read without a header, so we define the column names manually
CSV_PATH = "data/colors.csv"
df = pd.read_csv(CSV_PATH, names=["color", "color_name", "hex", "R", "G", "B", "tags"], header=None)

# Convert the RGB values from the CSV to a NumPy array for fast comparison
df_colors = df[["R", "G", "B"]].values.astype(int)
df_names = df["color_name"].values

def remove_background(image_path, output_folder="processed_images"):
    """
    Removes the background from an image using the rembg library and saves the result.
    
    Parameters:
        image_path (str): Path to the input image.
        output_folder (str): Folder where the processed image will be saved.
        
    Returns:
        str: The path to the processed image with no background.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the image and convert it to RGBA (to ensure an alpha channel is present)
    image = Image.open(image_path).convert("RGBA")
    
    # Remove the background using rembg
    output = remove(image)
    
    # Build the output file path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    processed_image_path = os.path.join(output_folder, f"{image_name}_no_bg.png")
    
    # Save the image without background
    output.save(processed_image_path)
    
    return processed_image_path

def remove_faces(image_path, output_folder="processed_images"):
    """
    Detects faces in the image and makes them transparent.
    
    Parameters:
        image_path (str): Path to the image.
        output_folder (str): Folder where the processed image will be saved.
        
    Returns:
        str: The path to the processed image with faces removed.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the image with OpenCV, preserving any alpha channel
    image_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image_cv is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # If the image does not have an alpha channel, add one
    if image_cv.shape[2] == 3:
        b, g, r = cv2.split(image_cv)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Create a full-opacity alpha channel
        image_cv = cv2.merge((b, g, r, alpha))

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image_cv[:, :, :3], cv2.COLOR_BGR2GRAY)
    
    # Load the Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # For each detected face, set the alpha channel to 0 (fully transparent)
    for (x, y, w, h) in faces:
        image_cv[y:y+h, x:x+w, 3] = 0

    # Build the output file path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    processed_face_path = os.path.join(output_folder, f"{image_name}_no_faces.png")
    
    # Save the processed image
    cv2.imwrite(processed_face_path, image_cv)

    return processed_face_path

def get_color_name(R, G, B):
    """
    Finds the closest color name in the CSV using Euclidean distance in RGB space.
    
    Parameters:
        R, G, B (int): The RGB values of the color to match.
    
    Returns:
        str: The name of the closest matching color.
    """
    # Calculate Euclidean distance from the input color to all reference colors
    distances = np.sqrt((df_colors[:, 0] - R) ** 2 + (df_colors[:, 1] - G) ** 2 + (df_colors[:, 2] - B) ** 2)
    
    # Return the name corresponding to the smallest distance (closest color)
    return df_names[np.argmin(distances)]

def extract_top_colors(image_path, num_colors=3):
    """
    Extracts the dominant colors from an image (with no background or faces) using KMeans clustering.
    
    Parameters:
        image_path (str): Path to the image.
        num_colors (int): Number of dominant colors to extract.
        
    Returns:
        list of str: List of detected color names.
    """
    # Read the image with OpenCV, preserving any alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # If the image has transparency, analyze only the visible (non-transparent) pixels
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        mask = a > 0
        # Rearrange the channels to RGB order for analysis
        pixels = np.array([r[mask], g[mask], b[mask]]).T
    else:
        # Reshape the image into a list of pixels
        pixels = np.reshape(img, (-1, 3))

    # If no pixels remain after filtering (unlikely), return "Unknown"
    if len(pixels) == 0:
        return ["Unknown"]

    # Apply KMeans clustering to find the dominant colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Assign names to the detected dominant colors using the get_color_name function
    detected_colors = [get_color_name(r, g, b) for r, g, b in dominant_colors]

    # Visualize the extracted dominant colors in a bar chart
    fig, ax = plt.subplots(figsize=(8, 2))
    for i, color in enumerate(dominant_colors):
        plt.bar(i, 1, color=np.array(color) / 255, width=1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Dominant Colors in the Image")
    plt.show()

    return detected_colors

def dominant_colors_to_tags(dominant_colors):
    """
    Given an array of dominant color names, this function looks up the corresponding tag
    in the DataFrame and returns an array of tags.
    
    Parameters:
        dominant_colors (list of str): Array of dominant color names.
    
    Returns:
        list of str: List of tags corresponding to each color.
    """
    color_tags = []
    
    # Loop through each dominant color
    for color in dominant_colors:
        # Search for a case-insensitive match in the "color_name" column
        match = df[df["color_name"].str.lower() == color.lower()]
        if not match.empty:
            # Retrieve the tag from the "tags" column
            tag = match.iloc[0]["tags"]
            color_tags.append(tag)
        else:
            color_tags.append(None)
    return color_tags

def process_image(image_path, num_colors=3, output_folder="processed_images"):
    """
    Pipeline that:
       1. Removes the background from the image.
       2. Removes faces from the image.
       3. Extracts the dominant colors from the processed image.
    
    Parameters:
        image_path (str): Path to the input image.
        num_colors (int): Number of dominant colors to extract.
        output_folder (str): Folder where processed images will be saved.
    
    Returns:
        list of str: List of dominant color names detected in the image.
    """
    # Remove the background from the original image
    bg_removed_path = remove_background(image_path, output_folder)
    # Remove faces from the image with the background removed
    face_removed_path = remove_faces(bg_removed_path, output_folder)
    # Extract and return the dominant colors from the processed image
    top_colors = extract_top_colors(face_removed_path, num_colors=num_colors)
    
    return top_colors
