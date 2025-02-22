import cv2
import numpy as np

def detect_stock_changes(reference_image_path, current_image_path):
    """
    Detects missing or misplaced items on a retail shelf by comparing the current image with a reference image.
    """
    # Load images
    ref_img = cv2.imread(reference_image_path)
    curr_img = cv2.imread(current_image_path)
    
    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(ref_gray, curr_gray)
    
    # Apply threshold to highlight differences
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours of differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around detected differences
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(curr_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Show results
    cv2.imshow("Differences Detected", curr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
reference_image = "shelf_full.jpg"  # Reference image of a fully stocked shelf
current_image = "shelf_current.jpg"  # Current shelf image to analyze
detect_stock_changes(reference_image, current_image)
