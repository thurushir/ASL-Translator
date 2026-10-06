# ASL-Translator
Real-Time ASL Translator
A computer vision project that recognizes American Sign Language (ASL) letters in real time using **MediaPipe Hands** and a **Random Forest classifier**.

## 🎥 Demo

<img src="FINAL_DEMO.gif" width="600" alt="ASL Translator Demo">
*The system performs real-time letter classification with confidence overlay.*

##  Features
- **Real-time hand tracking** using MediaPipe  
-  **Random Forest model** trained on 4,449 labeled samples  
- **Cross-validation accuracy:** 92.22% ± 0.76%  
-  Supports all **static ASL letters (A–Y except J, Z)**  
- **Smooth color-coded confidence gradient** display  
-  Modular code for easy model retraining or feature expansion  

> **Branch `temporalwip`:** work in progress on continuous fingerspelling. See [docs/TEMPORAL.md](docs/TEMPORAL.md).

## Training Summary
During training, the dataset contained a total of 4,449 labeled samples of ASL hand gestures. Out of these, 3,559 samples were used for training the model and 890 samples were reserved for testing.
After training, the Random Forest classifier achieved a single-split test accuracy of *93.15%*, showing strong performance on unseen data.
To further evaluate consistency, five-fold cross-validation was performed, resulting in an average accuracy of *92.22%* with a standard deviation of* 0.76%*, indicating that the model performs reliably across different data splits.

## Project Structure
```
ASL-Translator/
├── data/ # Labeled landmark data (by letter)
├── data_seq/ # Fingerspelling clips as landmark sequences (temporal work)
├── docs/TEMPORAL.md # Notes on the temporal (continuous fingerspelling) work
├── hand_utils.py # Helper functions for Mediapipe setup
├── features.py # Shared landmark feature extraction (frames and sequences)
├── transcript.py # Keeps predicted letters on screen as text
├── collect_sequences.py # Records fingerspelled words as landmark sequences
├── check_sequences.py # Summarizes recorded sequence clips
├── check_features.py # Verifies feature extraction matches the original
├── test_transcript.py # Tests the on-screen text rules
├── baseline_results.json # Static model accuracy (reference baseline)
├── process_data.py # Converts landmarks to training features
├── check_processed.py #Enumerates processed images per letter
├── train_model.py # Trains the Random Forest classifier
├── predict_live.py # Real-time prediction script
├── confusion_matrix.png #Confusion matrix of trained model
├── model.pickle # Saved trained model
├── data.pickle # Processed dataset
├── requirements.txt # Python dependencies with version constraints
├── demo.gif # Demo GIF (embedded above)
└── README.md
```

##  How to Run the ASL Translator
Follow these steps to get the ASL Translator working on your machine.

### Clone the Repository
```
git clone https://github.com/<thurushir>/ASL-Translator.git
cd ASL-Translator 
```

### Create a virtual environment and activate it 
```
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Install Dependencies
Install packages with version constraints to avoid compatibility issues between NumPy 2.0 and scikit-learn:
```
pip install -r requirements.txt
```
Alternatively, install manually:
```
pip install "numpy<2.0" "opencv-python<4.9" "scikit-learn>=1.3.0,<1.4" "mediapipe>=0.10.0" "matplotlib"
```
### Ensure these files exist in your project folder:
```
ASL-Translator/
├── model.pickle          # Trained Random Forest model
├── data.pickle           # Processed dataset (landmarks + labels)
├── hand_utils.py         # Mediapipe helper functions
├── predict_live.py       # Live ASL recognition script
├── process_data.py       # Converts hand landmarks into feature data
├── train_model.py        # Trains a new Random Forest model
```
If model.pickle or data.pickle are missing, generate them by running:
```
python process_data.py
python train_model.py
```
### Run it !
```
python predict_live.py
```

## Author
Thurushi Rajapakse
October 2025

This project was inspired by the article *“Sign Language Recognition using MediaPipe and Random Forest — Intel oneAPI Optimised Scikit-Learn Library”* by Vatika Agrawal.  
Read it here: [https://medium.com/@ag.vatika17/sign-language-recognition-using-mediapipe-and-random-forest-intel-oneapi-optimised-scikit-learn-f9e5b645aae2]
