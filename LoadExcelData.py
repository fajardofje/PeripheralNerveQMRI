#Script to load multiparametric PNS disease data
#and sepparate 3 classes: CMT1, CMT2 and Control
#data and sepparate them with multiparametric
#qMRI data using a SVM and generate ROC
#curves to assess the sepparation power
#01/27/2025
#Jesus Fajardo
#jesuseff@wayne.edu
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the Excel file
file_path = 'TestDataSet.xlsx'
excel_data = pd.read_excel(file_path)

# Filter data based on Phenotype
CMT1_data = excel_data[excel_data['Phenotype'] == 'CMT1']
CMT2_data = excel_data[excel_data['Phenotype'] == 'CMT2']
Control_data = excel_data[excel_data['Phenotype'] == 'Control']

# Combine the data for SVM training
combined_data = pd.concat([CMT1_data, CMT2_data, Control_data])

# Extract features (columns 6 to 15) and labels (Phenotype)
X = combined_data.iloc[:, 12:13].to_numpy()
print('Training set')
print(X)
y = combined_data['Phenotype'].to_numpy()

# Handle missing values by imputing with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.3, random_state=42)

# Train the SVM model
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Predict probabilities for the test set
y_prob = svm_model.predict_proba(X_test)

# Function to plot ROC curves
def plot_roc_curve(y_test, y_prob, class1, class2):
    # Filter test data for the two classes
    idx = np.where((y_test == class1) | (y_test == class2))
    y_test_filtered = y_test[idx]
    y_prob_filtered = y_prob[idx][:, [class1, class2]]

    # Binarize the labels for ROC curve calculation
    y_test_binary = np.where(y_test_filtered == class1, 1, 0)
    y_prob_binary = y_prob_filtered[:, 0]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test_binary, y_prob_binary)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {le.classes_[class1]} vs {le.classes_[class2]}')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curves for CMT1 vs Control, CMT2 vs Control, and CMT1 vs CMT2
plot_roc_curve(y_test, y_prob, le.transform(['CMT1'])[0], le.transform(['Control'])[0])
plot_roc_curve(y_test, y_prob, le.transform(['CMT2'])[0], le.transform(['Control'])[0])
plot_roc_curve(y_test, y_prob, le.transform(['CMT1'])[0], le.transform(['CMT2'])[0])
