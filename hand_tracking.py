import os
import pickle
import mediapipe as mp
import cv2 #computer's bridge to webcam
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands #gives accesss to MediaPipe's hand-detection model
mp_drawings =  mp.solutions.drawing_utils #gives helper fctns to draw landmarks on image
hands = mp_hands.Hands(
    static_image_mode=False, #expects live video stream and not single photos
    max_num_hands =1, #only detect one hand per frame
    min_detection_confidence = .5
)
#create video capturing object, acts like a connection to your camera
vid_cap = cv2.VideoCapture(0) #0 is default cam
while True:
    ret, frame = vid_cap.read() #ret returns T if frame was successfully captured, F ow., frame is the image NumPy arrays of pixel values
    if not ret:
        break
    frame = cv2.flip(frame,1) #flip the frame horizantally like a mirror
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# OpenCV captures in BGR; convert to RGB so MediaPipe reads colors correctly
    res = hands.process(frame_rgb) #asks Mediapipe to process the frame and look for hands
    #if at least one hand was found
    if res.multi_hand_landmarks: 
        for hand_landmarks in res.multi_hand_landmarks:
            #draw circles and connecting lines on your hand
            mp_drawings.draw_landmarks( 
                frame, #draw on the single frame
                #hand_landmarks consist or 21 points cpaturing the position of the wrist all five gfingers and all of the joints that 
                #  define the hand's shape and gesture. Each point consisits of x,y,z coordinates (x= l/r, y= high/low, z= depth)
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, #built-in list of which points to connect, like a stick figure hand
                mp_drawings.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3), #pink points for fun
                mp_drawings.DrawingSpec(color=(0, 255, 255), thickness=2) #yellow connections       
            )
    cv2.imshow("ASL Hand Detection", frame)

    #cv2.waitKey: waits 1 msec for a key presss, if key is pressed returns int code
    #0xFF: is bitwise operation to only look at last 8 bits, the ASCII key code checks if q was pressed then break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid_cap.release() #turns off the webcam, if not called webcame may stay locked 
cv2.destroyAllWindows() #close the OpenCV windowexit
