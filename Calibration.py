"""This script enables you to calibrate your dartboard (ie setting the 4 calibration points manually) 
before starting, to ensure they are correct and don't induce score prediction errors"""

import cv2
import os
import time
from datetime import datetime
import numpy as np
from time import time
from PIL import Image

def set_calibration_points(image_path):
    """Permet de calibrer manuellement la cible."""
    global CALIBRATION_POINTS
    CALIBRATION_POINTS = []

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = param.shape[:2]
            CALIBRATION_POINTS.append([x / w, y / h])
            print(f"Point ajouté : {[x / w, y / h]}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée : {image_path}")

    cv2.namedWindow("Définir les points de calibration")
    cv2.setMouseCallback("Définir les points de calibration", on_mouse_click, param=image)

    print("Cliquez sur les 4 points de calibration dans l'image.")
    while len(CALIBRATION_POINTS) < 4:
        temp_image = image.copy()
        for idx, point in enumerate(CALIBRATION_POINTS):
            px, py = int(point[0] * temp_image.shape[1]), int(point[1] * temp_image.shape[0])
            cv2.circle(temp_image, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(temp_image, f"P{idx+1}", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Définir les points de calibration", temp_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processus annulé.")
            CALIBRATION_POINTS = []
            cv2.destroyAllWindows()
            break
    print("Tous les points de calibration sont définis.")
    cv2.destroyAllWindows()
    return CALIBRATION_POINTS


def capture_image(): 
    save_folder = "live_demo_photos"
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize the camera
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Set the camera resolution to 800 x 800
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {int(width)}x{int(height)}")

    ret, frame = cap.read()
    if ret:
        # Generate a filename using the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_folder, f"photo_{timestamp}.jpg")

        # Save the captured frame as an image file
        cv2.imwrite(filename, frame)
        print(f"Photo saved as {filename}")
    else:
        print("Error: Could not capture photo.")
    # Release the camera
    cap.release()
    return filename

def crop_and_display(image_path, left_crop, right_crop, top_crop, bottom_crop):
    with Image.open(image_path) as img:
        width, height = img.size
        cropped = img.crop((left_crop, top_crop, width - right_crop, height - bottom_crop))
        resized_image = cropped.resize((800, 800))
        resized_image.save(image_path)


if __name__ == "__main__":
    filename = capture_image()
    crop_and_display(filename,left_crop=0, right_crop=0, top_crop=0, bottom_crop=0) #utiliser crop si besoin
    # Définir les points de calibration
    CALIBRATION_POINTS = set_calibration_points(filename)
    if isinstance(CALIBRATION_POINTS, list):
        # Convertir en tableau NumPy
        CALIBRATION_POINTS = np.array(CALIBRATION_POINTS, dtype=np.float32)  
    # Ajouter une colonne de 1 (confidence score) pour être compatible avec le format de prédiction de deep-darts
    CALIBRATION_POINTS = np.hstack((CALIBRATION_POINTS, np.ones((CALIBRATION_POINTS.shape[0], 1))))
    np.save("calibration_points.npy", CALIBRATION_POINTS)   