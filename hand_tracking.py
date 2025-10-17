import os
import pickle
import mediapipe as mp
import cv2 #computer's bridge to webcam
import matplotlib.pyplot as plt
import hand_utils as h


hands = h.init_hands()

h.start_video(hands, h.draw_hand_landmarks)