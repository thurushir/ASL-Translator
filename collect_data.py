import cv2
import mediapipe as mp
import os
import time
import hand_utils as h

#set up Data Folder
DIR = "./data" #where captured images will go
if not os.path.exists(DIR):
    os.makedirs(DIR)

hands = h.init_hands()

#makes folder in DIR for letter
letter = input("Input ASL letter to collect data for:").upper() 
save_path = os.path.join(DIR, letter)
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"\nCollecting data for letter: {letter}")
print("Press 'q' to quit.\n")

DATASET_SIZE = 100   # how many images to save automatically
SAVE_INTERVAL = 0.1  # seconds between saved frames 
counter = 0          # track number of images saved
last_save_time = 0   # keeps track of time since last save

#Countdown 3,2,1
cap = cv2.VideoCapture(0)
for i in range(3, 0, -1):
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Starting in {i}...", (200, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.imshow("ASL Hand Detection", frame)
    cv2.waitKey(1000)
cv2.destroyAllWindows()
cap.release()

def handle_frame(frame, results):
    global counter, last_save_time
    h.draw_hand_landmarks(frame, results)

    #Displpays Letter 
    cv2.putText(frame, f'Letter: {letter} | Saved: {counter}/{DATASET_SIZE}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # get current time to throttle saves (avoid saving 60 fps = too fast)
    current_time = time.time()
    if current_time - last_save_time >= SAVE_INTERVAL and counter < DATASET_SIZE:
        path = os.path.join(save_path, f'{counter}.jpg')
        cv2.imwrite(path, frame) #cv2.imwrite writes an image to disk, [path] is where the [frame] it should be saved
        print(f"Saved {path}")
        counter += 1
        last_save_time = current_time
    if counter >= DATASET_SIZE:
        print(f"Finished collecting {DATASET_SIZE} images for {letter}.")
        cv2.destroyAllWindows()
        os._exit(0)  # end cleanly      
h.start_video(hands, handle_frame)
  

    
      