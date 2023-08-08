import pytesseract
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os
import cv2

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Images dir
images_dir = r"./img"

# Function to handle mouse events
def get_coordinates(event, x, y, flags, param):
    global image, clicked_points, bounding_boxes, angle
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates: ({x}, {y})")
        clicked_points.append((x, y))
        
        if len(clicked_points) == 4:
            # Draw a custom path
            cv2.polylines(image, [np.array(clicked_points)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("Image", image)

            # Store bounding box coordinates
            bounding_boxes.append((clicked_points, angle, len(bounding_boxes) + 1))
            clicked_points = []

# Loop through images in the folder
for image_file in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # Set a 4K window size (3840x2160)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 3840, 2160)

    # List to store the clicked points, bounding boxes, and rotation angle
    clicked_points = []
    bounding_boxes = []
    angle = 0  # Initialize the angle to 0 degrees

    # Set the mouse callback function
    cv2.setMouseCallback("Image", get_coordinates)

    while True:
        # Display the image
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        # Rotate the image clockwise
        if key == ord("r"):
            angle -= 10

        # Rotate the image counterclockwise
        if key == ord("R"):
            angle += 10

        # Save the bounding boxes and move to the next image
        if key == ord("n"):
            if bounding_boxes:
                for box in bounding_boxes:
                    custom_path = np.array(box[0])
                    rotated_box = (custom_path, box[1] + angle, box[2])
                    
                    # Crop the custom region
                    mask = np.zeros_like(image)
                    cv2.fillPoly(mask, [custom_path], (255, 255, 255))
                    masked_image = cv2.bitwise_and(image, mask)
                    bbx_name = f"bbx_{rotated_box[2]}_{os.path.basename(image_path)}"
                    cv2.imwrite(os.path.join(images_dir, bbx_name), masked_image)
                    custom_cropped_image = Image.fromarray(masked_image)
                    text = pytesseract.image_to_string(np.array(custom_cropped_image))
                    print(f"Detected Text in {bbx_name} (Box {rotated_box[2]}): {text}")
                bounding_boxes = []
            break

        # Exit
        elif key == 27:  # Esc key
            for box in bounding_boxes:
                custom_path = np.array(box[0])
                rotated_box = (custom_path, box[1] + angle, box[2])
                mask = np.zeros_like(image)
                cv2.fillPoly(mask, [custom_path], (255, 255, 255))
                masked_image = cv2.bitwise_and(image, mask)
                custom_cropped_image = Image.fromarray(masked_image)
                print(f"Box {rotated_box[2]}: Detected Text: {pytesseract.image_to_string(np.array(custom_cropped_image))}")
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()
