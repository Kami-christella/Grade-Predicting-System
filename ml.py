import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Use raw string to specify the file path
file_path = r"C:\Users\Kami\Desktop\AUCA STUDIES\Big Data\Exam preparation\two\project1\Data.xlsx"

# Load the dataset
try:
    dataset = pd.read_excel(file_path)
    print("Dataset loaded successfully.")
    print(dataset.describe())
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()

# 1. Remove duplicates
dataset.drop_duplicates(inplace=True)

# 2. Handle null values
meanOfQuizzes = dataset["QUIZZES "].mean()
dataset.fillna({"QUIZZES ": meanOfQuizzes}, inplace=True)

# 3. Handle wrong data formats (if any) - specific examples needed
# 4. Correct wrong data values
for x in dataset.index:
    if dataset.loc[x, "QUIZZES "] < 0:
        dataset.loc[x, "QUIZZES "] = 0
    if dataset.loc[x, "QUIZZES "] > 30:
        dataset.loc[x, "QUIZZES "] = 30

# Prepare the data for machine learning
X = dataset.drop(columns=['SNAMES ', 'Total Marks', 'Marks /20', 'Grading '])
y = dataset['Grading ']

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize models
Decision_tree_model = DecisionTreeClassifier()
Logistic_regression_Model = LogisticRegression(solver='lbfgs', max_iter=10000)
SVM_model = svm.SVC(kernel='linear')
RF_model = RandomForestClassifier(n_estimators=100)

# Train models
Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)
RF_model.fit(X_train, y_train)

# Predict with models
DT_Prediction = Decision_tree_model.predict(x_test)
LR_Prediction = Logistic_regression_Model.predict(x_test)
SVM_Prediction = SVM_model.predict(x_test)
RF_Prediction = RF_model.predict(x_test)

# Evaluate models
DT_score = accuracy_score(y_test, DT_Prediction)
LR_score = accuracy_score(y_test, LR_Prediction)
SVM_score = accuracy_score(y_test, SVM_Prediction)
RF_score = accuracy_score(y_test, RF_Prediction)

print("Decision Tree accuracy =", DT_score * 100, "%")
print("Logistic Regression accuracy =", LR_score * 100, "%")
print("Support Vector Machine accuracy =", SVM_score * 100, "%")
print("Random Forest accuracy =", RF_score * 100, "%")

# Make a prediction
predict = Logistic_regression_Model.predict([[2, 27, 0, 15]])
print("Prediction:", predict)

# Save the model
joblib.dump(Logistic_regression_Model, 'abc.joblib')
