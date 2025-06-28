import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report

#P and C, under Category in the dataset means Passengers and C

esto_data= pd.read_csv("estonia-passenger-list.csv")



X = esto_data[['Country','Sex', 'Age' , 'Category' ]].copy()
Y = esto_data["Survived"]



#changed string to numeric form 
X['Sex']= X['Sex'].map({'M': 0 , 'F': 1})

#it uses one_hot encoding method , to change the strings  to numeric data, to make it possible to train the model
X = pd.get_dummies(X, columns=['Country', 'Category'])

X['Age']= X['Age'].fillna(X['Age'].median())

#we set a "random state " value to any arbitrary value, so that data becomes reproducible
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=11)

#this way we get to test the models accuracy, even when we add new feature

rf_model= RandomForestClassifier( n_estimators=100,random_state=11)
#n_estimators,is the number of decision trees to create

rf_model.fit(X_train, Y_train)
#it tells the rfmodel 'rf_classifier' , to learn(fit) patterns from those parameters 

Y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)


sample =  X_test.sample(n=1, random_state=None) #selects a random row ,random_state=None for different random rows each time
prediction = rf_model.predict(sample)

# Get the index of that sample
sample_index = sample.index[0]

# Get the actual survival status from Y_test
actual = Y_test.loc[sample_index]


sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Passenger: {sample_dict}")
print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
print(f"Actual Survival: {'Survived' if actual == 1 else 'Did Not Survive'}")






        




