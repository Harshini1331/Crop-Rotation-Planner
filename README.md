# Crop-Rotation-Planner
This project is a Crop Rotation Planner designed to assist farmers in optimizing their crop rotations for sustainable and efficient farming practices. It provides recommendations for crop sequences based on various factors such as soil health, environmental conditions, and nutrient balance.

## Features

- Recommends crop rotation plans based on user input and historical data.
- Takes into account soil health, nutrient levels, and climatic conditions.
- Provides visual representations of data.
- Allows users to customize and prioritize specific crops in their rotation.

## Requirements

- Python 3.x
- Required Python packages: streamlit, pandas, numpy, matplotlib, scikit-learn

## Architecture Diagram/Flow

<img width="623" alt="1" src="https://github.com/Harshini1331/Crop-Rotation-Planner/assets/75235554/8d94240d-d4d8-4b70-bcfb-0130ba88364e">

## Program:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

# Load the CSV file data into data variable using pandas
PATH = 'Crop_recommendation.csv'
data = pd.read_csv(PATH)

data.head()
data.info()
data.describe()
data['label'].unique()
data['label'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 60

features = ['N', 'P', 'K', 'temperature',
            'humidity', 'ph', 'rainfall']

for i, feat in enumerate(features):
    plt.subplot(4, 2, i + 1)
    sns.histplot(data[feat], color='greenyellow', kde=True)  # You can use sns.histplot for histograms
    if i < 3:
        plt.title(f'Ratio of {feat}', fontsize=12)
    else:
        plt.title(f'Distribution of {feat}', fontsize=12)
    plt.tight_layout()
    plt.grid()

plt.show()

fig, ax = plt.subplots(1, 1, figsize=(15, 9))
sns.heatmap(data.corr(),
			annot=True,
			cmap='viridis')
ax.set(xlabel='features')
ax.set(ylabel='features')

plt.title('Correlation between different features',
		fontsize=15,
		c='black')
plt.show()

# Put all the input variables into features vector
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Put all the output into labels array
labels = data['label']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, Y_train)
dt_predicted_values = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)
rf_predicted_values = rf_model.predict(X_test)

# Support Vector Machine (SVM)
svm_model = SVC(random_state=42)
svm_model.fit(X_train, Y_train)
svm_predicted_values = svm_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, Y_train)
lr_predicted_values = lr_model.predict(X_test)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, Y_train)
knn_predicted_values = knn_model.predict(X_test)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)
nb_predicted_values = nb_model.predict(X_test)


# Evaluate and compare the models
models = {
    'Decision Tree': dt_predicted_values,
    'Random Forest': rf_predicted_values,
    'SVM': svm_predicted_values,
    'Logistic Regression': lr_predicted_values,
    'KNN': knn_predicted_values,
    'Naive Bayes': nb_predicted_values
}

best_model_name = None
best_accuracy = 0.0

for model_name, predictions in models.items():
    accuracy = accuracy_score(Y_test, predictions)
    print(f"{model_name} accuracy: {accuracy}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name

if best_model_name == 'Decision Tree':
    best_model = dt_model
elif best_model_name == 'Random Forest':
    best_model = rf_model
elif best_model_name == 'SVM':
    best_model = svm_model
elif best_model_name == 'Logistic Regression':
    best_model = lr_model
elif best_model_name == 'KNN':
    best_model = knn_model
elif best_model_name == 'Naive Bayes':
    best_model = nb_model

# Streamlit App
st.title("Crop Rotation Planner")

# User Input for Features
st.header("Enter the Input Features")
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, step=1.0)
P = st.number_input("Phosphorous (P)", min_value=0.0, max_value=150.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=150.0, step=1.0)
temperature = st.number_input("Temperature", min_value=0.0, max_value=100.0, step=1.0)
humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, step=1.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall", min_value=0.0, max_value=300.0, step=1.0)

# Predict using the best model
if st.button("Predict"):
    new_data_values = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = best_model.predict(new_data_values)
    st.success(f"The predicted crop label is {prediction[0]}")
```

## Output:

![image](https://github.com/Harshini1331/Crop-Rotation-Planner/assets/75235554/ab2c3cc2-8cfc-43a1-b092-4c8f94b92d91)
![image](https://github.com/Harshini1331/Crop-Rotation-Planner/assets/75235554/d572d5e9-1fe6-4532-81ba-733fd0edaae1)
![image](https://github.com/Harshini1331/Crop-Rotation-Planner/assets/75235554/97cb2746-ce83-4da1-bb8c-656e8ff4205e)
![image](https://github.com/Harshini1331/Crop-Rotation-Planner/assets/75235554/a5a2c483-7c79-47f4-8076-c1172227e132)
![image](https://github.com/Harshini1331/Crop-Rotation-Planner/assets/75235554/d31cfdca-d352-44f8-a5e7-e2fd27c2f093)
![image](https://github.com/Harshini1331/Crop-Rotation-Planner/assets/75235554/732d85a6-d1ea-4eeb-b577-a9a2f65d2eb2)

## Result:

The integration of diverse machine learning models, including Decision Trees, Random Forest, SVM, Logistic Regression, K-Nearest Neighbors, and Naive Bayes, has resulted in precise and data-driven crop rotation recommendations. This achievement enhances the efficiency of agricultural planning and resource allocation.

This Crop Rotation Planner project provides farmers with data-driven recommendations for optimal crop rotations. By leveraging machine learning models, it takes into account various factors to suggest rotations that enhance soil health and overall agricultural productivity.
