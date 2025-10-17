import os
import pickle
import cv2
import mediapipe as mp
import hand_utils as h

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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #OpenCV reafs as BGR, MediaPipe expects RGB so need to reorder
        res= hands.process(img_rgb) #send image through MP hands model, and detects landmarks
        
        #if hands detected, extract landmark data
        if res.multi_hand_landmarks:
            for h_landmark in res.multi_hand_landmarks:
                x = [lm.x for lm in h_landmark.landmark]
                y = [lm.y for lm in h_landmark.landmark]
                z = [lm.z for lm in h_landmark.landmark]

                norm_hand= []
                #normalize to make gesture independent of hand's screen position and camea distance
                #normalize the hand's position, x,y to top-left and z relative to wrist and storing it back in a flattened array [norm_hand]
                for lm in h_landmark.landmark:
                    norm_hand.append(lm.x-min(x)) 
                    norm_hand.append(lm.y-min(y))
                    norm_hand.append(lm.z- h_landmark.landmark[0].z)
                data.append(norm_hand)
                letters.append(folder)
# cv2.destroyAllWindows()


#save data set 
#use "with" so the file auto closes after done safely even if an error happenss
with open('data.pickle', 'wb') as f: #open a new file in binary mode since pickle doesn't save plain text, only serialzied Python objects
    pickle.dump({'data': data, 'letters': letters}, f) #"dump" writes the object into the file

print(f"\nProcessed {len(data)} total samples.")
print(f" Saved 3D (x, y, z) landmark dataset to 'data.pickle'")


            
                        




