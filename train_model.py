import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

data_dict = pickle.load(open('data.pickle', 'rb')) #opens file in read binary mode
data = data_dict['data']
letters = data_dict['letters']
print(f"Loaded {len(data)} sampels from data.pickle")

#split data, x being the landmark patterns and y being the corresponding letters
#we set aside 20 percent for testing 
#stratify=letters ensure all letters are represented equally in both sets
x_train, x_test, y_train, y_test = train_test_split(data, letters, test_size = .2, shuffle= True, stratify=letters)

print(f"Training samples: {len(x_train)} | Testing samples: {len(x_test)}")

#intialize model:
#Random Forest Classifier is an ensemble model being a series decision trees, each learning patterns slightly differently, final prediction is majority vote of all trees
#Each tree randomly selects a subset of your training samples and features (landmark coordinates) and trains a DT (bootstrap sampling)
model = RandomForestClassifier(
    n_estimators=100, #number of trees, more inc accuracy but slower. 
    random_state =42, #keeps results reproducable every time training: seed for the random number generator used, 42 is default constant
    n_jobs = -1 #uses ALL CPU cores for speed (parallel processing)
)

print("Training the Random Forest Model...")
#train the model
model.fit(x_train, y_train) 
print("Finished training!")


predictions = model.predict(x_test)

#evaluate on a single test set
single_acc = accuracy_score(y_test, predictions)
print(f"Single Split Test Accuracy: {single_acc*100:.2f}%")

#evaluate on test set: 7-Fold Cross-Validation
# Splits the entire dataset into 7 parts; trains on 6 and tests on the 7th, repeating this process 7 times.
# The average of these 7 accuracies gives a more reliable estimate of performance than one random train/test split.
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

cv = KFold(n_splits=7, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, data, letters, cv=cv, n_jobs=-1)

print(f"Cross-Validation Accuracies: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores)*100:.2f}%")
print(f"Standard Deviation: {np.std(cv_scores)*100:.2f}%")


# Confusion Matrix: Visualizes how well the model predicts each ASL letter.
# Rows = true labels, Columns = predicted labels.
# Bright diagonal cells → correct predictions.
# Off-diagonal cells → misclassifications (e.g., model confused one letter for another).
# The color bar shows how many samples fall into each cell.
cm = confusion_matrix(y_test, predictions, labels=sorted(set(y_test)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(y_test))) #visualize
disp.plot(xticks_rotation='vertical')
plt.title("ASL Letter Classification Confusion Matrix")
plt.tight_layout()
print("Press 'q' to close window or 's' to save it as a png, and then hit Enter in  terminal.'")

plt.show(block=False)
while True:
    user_input = input().strip().lower()
    if user_input =="s":
        plt.savefig("confusion_matrix.png", dpi=300)
        print("Saved plot as 'confusion_matrix.png'")
    elif user_input == 'q':
        plt.close('all')
        break

with open('model.pickle', 'wb') as f:
    pickle.dump({'model': model}, f) #save 
print("Saved trained model as model.pickle")
