import os
import pickle
import cv2
import hand_utils as h
import features as F
import numpy as np
import random

def augment_image(img):
    h, w = img.shape[:2] #img.shape returns (heights, width, channels)

    # Random rotation (±5°)
    angle = random.uniform(-5, 5)
    #Generates a 2D affine rotation matrix used by OpenCV to rotate the image. The middle of the image (w / 2, h / 2) is the rotation center. 1, is the scale-factor, none.
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1) 
    img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random brightness (±20%). Multiplies each pixel by the brightness factor.  
    #np.clip(..., 0, 255) ensures pixel values stay valid (between 0 and 255), to avoid overflow
    #.astype(np.uint8)` converts back to standard 8-bit integers (the format OpenCV uses).
    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    return img

#initalize Mediapipe hands in static image mode (one image at a time) since we will be working with saved photos **
hands = h.init_hands(static_mode=True)

DIR = "./data"
data = [] #stores flattened landmark coordinates (21 hand landmarks with (x,y,z) turns into one tuple. Random Forests exepect each trianing sample to be a flat vector)
letters = []

#loop through each letter folder
for folder in os.listdir(DIR): #os.listdir list every item in data/ dir
    print(f"Processing folder: {folder}")
    folder_path = os.path.join(DIR, folder)
    if not os.path.isdir(folder_path): #check if actually a folder and not a stray file (which we should skip over)
        continue

    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        img = cv2.imread(img_path) #OpenCV reads the image into memory as a NumPy array (3D array of pxls), if it cant read a file returns none, uses BGR order
        if img is None:
            continue
        imgs_to_process = [img, augment_image(img)] #image and augmented image 
        for current_img in imgs_to_process:
            img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB) #OpenCV reafs as BGR, MediaPipe expects RGB so need to reorder
            res= hands.process(img_rgb) #send image through MP hands model, and detects landmarks
        
            #if hands detected, extract normalized landmark features (see features.py for details)
            norm_hand = F.frame_features(res)
            if norm_hand is not None:
                data.append(norm_hand.tolist())
                letters.append(folder)
# cv2.destroyAllWindows()


#save data set 
#use "with" so the file auto closes after done safely even if an error happenss
with open('data.pickle', 'wb') as f: #open a new file in binary mode since pickle doesn't save plain text, only serialzied Python objects
    pickle.dump({'data': data, 'letters': letters}, f) #"dump" writes the object into the file

print(f"\nProcessed {len(data)} total samples.")
print(f" Saved 3D (x, y, z) landmark dataset to 'data.pickle'")


            
                        




