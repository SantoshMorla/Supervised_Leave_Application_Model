#नमः सिद्धं
from sklearn.model_selection import train_test_split
#import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv('D:\\wlc\\dataset.csv') # read the csv file and store it in a dataframe

data['PDD'] = pd.to_datetime(data['PDD']) # convert to datetime format
data['LD'] = pd.to_datetime(data['LD'])

data['Days_Diff'] = (data['PDD'] - data['LD']).dt.days # calculate how many days before the deadline

# Split data into features and labels
X = data['Days_Diff'].values.reshape(-1, 1) # feature: days before the deadline
y = data['AS'].values # label: approved or not

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) # split the data into 60% training and 40% test sets  
print(X_train, y_train)

# Train a logistic regression model
model = LogisticRegression() # create a logistic regression object
model.fit(X_train, y_train) # fit the model on the training data

# Predict on test data
y_pred = model.predict(X_test) # predict the labels using the model

# Evaluate the model performance
acc = accuracy_score(y_test, y_pred) # calculate the accuracy score

print("The accuracy score on the test data is:", acc)
#print()

# Print the status of approved and rejected status for each user
data['Predicted'] = model.predict(X) # add the predicted labels to the data
status = data.groupby(['Name', 'AS', 'Predicted']).size().unstack(fill_value=0) # group by name, actual and predicted labels and count the frequency
status.columns = ['Rejected', 'Approved'] # rename the columns
print("The status of approved and rejected status for each user is:")
print(status)