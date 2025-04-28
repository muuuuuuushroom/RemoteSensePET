import cv2
import os
import numpy as np
from tqdm import tqdm

def read_image(image_path, target_size=None):
    """Read an image from file and optionally resize it."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    if target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image

def find_most_similar_image(reference_image_path, folder_path, target_size=(1024, 1024)):
    """Find the most similar image to the reference image in the given folder using ORB features."""
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Read and prepare the reference image
    reference_image = read_image(reference_image_path, target_size)
    kp_ref, des_ref = orb.detectAndCompute(reference_image, None)

    max_similarity = -1
    most_similar_image_path = None

    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                try:
                    image = read_image(image_path, target_size)
                    if image is not None:
                        kp, des = orb.detectAndCompute(image, None)

                        if des is not None and des_ref is not None:
                            # Create BFMatcher object
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                            matches = bf.match(des_ref, des)
                            similarity = len(matches)  # Simple match count as a similarity score

                            if similarity > max_similarity:
                                max_similarity = similarity
                                most_similar_image_path = image_path
                except Exception as e:
                    print(f"Could not process {image_path}: {e}")

    return most_similar_image_path, max_similarity

# Example usage
if __name__ == "__main__":
    reference_image_path = '11.png'  # Replace with your reference image path
    folder_path = '/data/zlt/PET/RTC/data/Ship/Images'  # Replace with your folder path

    most_similar_image, similarity_score = find_most_similar_image(reference_image_path, folder_path)
    print(f"The most similar image is {most_similar_image} with a similarity score of {similarity_score}")