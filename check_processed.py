import pickle
from collections import Counter
import mediapipe as mp
import os
import cv2

# Load your dataset
data_dict = pickle.load(open('data.pickle', 'rb'))
letters = data_dict['letters']

# Count how many samples per letter
counts = Counter(letters)

# Sort from least to greatest
sorted_counts = sorted(counts.items(), key=lambda x: x[1])

print("\nLetter sample counts least to greatest):\n")
for rank, (letter, count) in enumerate(sorted_counts, start=1):
    print(f"{rank:2}. {letter}: {count} samples")

# Summary stats
total = sum(counts.values())
min_letter, min_count = sorted_counts[0]
max_letter, max_count = sorted_counts[-1]


print(f"Total samples: {total}")
print(f"Lowest count: '{min_letter}' ({min_count} samples)")
print(f" Highest count: '{max_letter}' ({max_count} samples)")
