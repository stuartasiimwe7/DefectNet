import numpy as np
import cv2

def apply_threshold(image, threshold=127):
    """
    Apply a binary threshold to the input image.
    
    Parameters:
    image (numpy.ndarray): Input image.
    threshold (int): Threshold value.
    
    Returns:
    numpy.ndarray: Thresholded image.
    """
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def find_contours(binary_image):
    """
    Find contours in a binary image.
    
    Parameters:
    binary_image (numpy.ndarray): Input binary image.
    
    Returns:
    list: List of contours found in the image.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image, contours):
    """
    Draw contours on an image.
    
    Parameters:
    image (numpy.ndarray): Input image.
    contours (list): List of contours to draw.
    
    Returns:
    numpy.ndarray: Image with contours drawn.
    """
    return cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

def postprocess_image(image_path, threshold=127):
    """
    Postprocess an image by applying thresholding and finding contours.
    
    Parameters:
    image_path (str): Path to the input image.
    threshold (int): Threshold value.
    
    Returns:
    numpy.ndarray: Image with contours drawn.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    binary_image = apply_threshold(image, threshold)
    contours = find_contours(binary_image)
    result_image = draw_contours(image, contours)
    
    return result_image