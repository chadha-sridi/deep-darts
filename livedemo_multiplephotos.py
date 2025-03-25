import cv2
import os
import time
from datetime import datetime
from yolopredict import bboxes_to_xy, load_model, predict
from yolov4.tf import YOLOv4
from dataset.annotate import get_dart_scores
import os.path as osp
import numpy as np
from tensorflow.keras.models import Model
from PIL import Image
import argparse
from yacs.config import CfgNode as CN
import keyboard

# Define the folder to save photos
def capture_image(): 
    save_folder = "live_demo_photos"
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 is the default camera
    #set camera resolution to 800 x 800
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

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
    """Crops and resizes the captured image."""
    with Image.open(image_path) as img:
        width, height = img.size
        cropped = img.crop((left_crop, top_crop, width - right_crop, height - bottom_crop))
        resized_image = cropped.resize((800, 800))
        resized_image.save(image_path)


def predict_2(input_image, calibration_points , model,cfg, max_darts = 3): 
        
    image_path = input_image
    img = cv2.imread(image_path)
   
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = model.predict(img_rgb)
    xy_pred = bboxes_to_xy(bboxes, max_darts)
    #overwrite the calibration points
    xy_pred[:4,:]=   calibration_points
    missing_idx = np.where(xy_pred[:4, -1] == 0)[0]

    if len(missing_idx) == 0:
        score = get_dart_scores(xy_pred, cfg, numeric = True)
        print(xy_pred)
        print(score)
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_d2')
    parser.add_argument('-w', '--weights_path', default=r'C:\Users\MSI\deep-darts\models\deepdarts_d3\100epochs\weights')
    args = parser.parse_args()

    # Load configuration
    config_path = 'configs/deepdarts_d2.yaml'
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg
    # Load the Calibration 
    calibration_points = np.load('calibration_points.npy')
    # Load the model
    yolo_model = load_model(config_path, args.weights_path)
    uppermodel = YOLOv4(tiny=True)
    uppermodel.classes = "classes"
    uppermodel.input_size = (800, 800)
    uppermodel.batch_size = 1
    uppermodel.model = yolo_model
    
    print("Press 's' to capture a photo and get a score. Press 'q' to quit.")

    while True:
        # If 's' is pressed, capture and process an image
        if keyboard.is_pressed('s'):  
            start_cap = time.time()
            filename = capture_image()
            end_cap = time.time() 
            print("time of capture", end_cap - start_cap )
            print(filename)
            if filename:
                #Adjust as needed accroding to your capture resolution
                crop_and_display(filename, left_crop=800, right_crop=230, top_crop=260, bottom_crop=0) 
                start_pred = time.time() 
                predict_2(filename,calibration_points, uppermodel, cfg)
                end_pred = time.time() 
                print("time of prediction", end_pred - start_pred)
        elif keyboard.is_pressed('q'):  
            print("Exiting...")
            break
    cv2.destroyAllWindows()
