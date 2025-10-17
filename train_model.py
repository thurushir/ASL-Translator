import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('data.pickle', 'rb')) #opens file in read binary mode
data = data_dict['data']
letters = data_dict['letters']
print(f"Loaded {len(data)} sampels from data.pickle")

#split data, x being the laandmark patterns and y being the corresponding letters
#we set aside 20 percent for testing 
#stratify=letters ensure all letters are represented equally in both sets
x_train, x_test, y_train, y_test = train_test_split(data, letters, tets_size = .2, shuffle= True, stratify=letters)

print(f"Training samples: {len(x_train)} | Testing samples: {len(y_train)}")

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

#evaluate on test set
predictions = model.predict(x_test)
acc= accuracy_score(y_test, predictions)
print(f"Model accuracy: {acc*100:.2f}%") #what is this

with open('model.pickle', 'wb') as f:
    pickle.dump({'model': model}, f) #save 
print("Saved trained model as model.pickle")
