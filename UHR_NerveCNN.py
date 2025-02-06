# -*- coding: utf-8 -*-
#Script to load UHR MRI image and the nerve
#segmentation

#02/05/2025
#Jesus Fajardo
#jesuseff@wayne.edu
'''
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################
###############################################
#LOADING AND PREPROCESSING DATA PART
def read_nifti(filepathname):
    nii = nib.load(filepathname)
    img = nii.get_fdata()
    dim = img.shape

    pixdim = [1, 1, 1]
    pixdim[0] = nii.header['pixdim'][1]
    pixdim[1] = nii.header['pixdim'][2]
    pixdim[2] = nii.header['pixdim'][3]

    if len(dim) == 3:
        newimg = np.zeros((dim[1], dim[0], dim[2]))
        for i in range(dim[2]):
            newimg[:, :, i] = np.flipud(img[:, :, i].T)
    elif len(dim) == 2:
        newimg = np.flipud(img.T)
    elif len(dim) == 4:
        newimg = np.zeros((dim[1], dim[0], dim[2], dim[3]))
        for ii in range(dim[3]):
            for i in range(dim[2]):
                newimg[:, :, i, ii] = np.flipud(img[:, :, i, ii].T)

    return newimg, dim, pixdim

def read_tob(fname, dim):
    with open(fname, 'rb') as f:
        num_obj = np.fromfile(f, dtype=np.int32, count=1)[0]  # always 0
        num_obj = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        mask = np.zeros(dim, dtype=np.float32)
        label = np.zeros(dim, dtype=np.float32)
        center = np.zeros((num_obj, 3), dtype=np.float32)
        extent = np.zeros((num_obj, 3), dtype=np.float32)
        flag = np.ones(num_obj, dtype=np.int32)
        
        for ii in range(num_obj):
            num_pts = np.fromfile(f, dtype=np.int32, count=1)[0]
            temp = np.fromfile(f, dtype=np.int32, count=num_pts)
            temp[temp < 1] = 0
            temp[temp > max(dim)] = 0
            
            xx = temp[0:num_pts:4]
            yy = temp[1:num_pts:4]
            zz = temp[2:num_pts:4]
            
            ind_valid = xx > 0
            if np.sum(ind_valid) == 0:
                flag[ii] = 0
                continue
            
            xx = xx[ind_valid] + 1  # objects are 0-started
            yy = yy[ind_valid] + 1
            zz = zz[ind_valid]
            
            indtemp = np.ravel_multi_index((yy, xx, zz), dim)
            mask.flat[indtemp] = 1
            label.flat[indtemp] = ii + 1
            
            center[ii, 0] = round((np.min(yy) + np.max(yy)) / 2)
            center[ii, 1] = round((np.min(xx) + np.max(xx)) / 2)
            center[ii, 2] = round((np.min(zz) + np.max(zz)) / 2)
            extent[ii, 0] = np.max(xx) - np.min(xx)
            extent[ii, 1] = np.max(yy) - np.min(yy)
            extent[ii, 2] = np.max(zz) - np.min(zz)
        
        num_obj = np.sum(flag)
        center = center[flag == 1, :]
        extent = extent[flag == 1, :]
    
    return mask, num_obj, center, extent, label

# Load the PatientID from the Excel file
excel_path = 'D:/ThighData.xlsx'
df_patients = pd.read_excel(excel_path)

# Define the base paths for MRI images and binary masks
base_mri_path = 'D:/CMT_Project_Results_Data_All/'
base_mask_path = 'D:/CMT_Project_ROI_Data/'

# Function to load and process MRI and mask data for a given patient index and slice index
def process_patient_data(patient_index, slice_index, crop_size=60):
    patient_id = df_patients.iloc[patient_index]['PatientID']
    
    mri_filepathname = os.path.join(base_mri_path, patient_id, 'thigh_final_UHR_corr.nii')
    mask_filepathname = os.path.join(base_mask_path, f'{patient_id}_thigh_UHR_Nerve.tob')
    
    # Load the MRI image (nii file)
    mri_data, mri_dim, mri_pixdim = read_nifti(mri_filepathname)

    # Load the mask image (tob file)
    mask_data, num_obj, center, extent, label = read_tob(mask_filepathname, mri_dim)

    # Ensure the mask and MRI data have the same shape
    if mri_data.shape != mask_data.shape:
        raise ValueError("The MRI image and mask image must have the same shape.")

    # Multiply each slice of the MRI data by the mask
    masked_mri_data = mri_data * mask_data

    # Locate the centroid of the mask in the middle slice
    centroid_y, centroid_x = np.argwhere(mask_data[:, :, slice_index]).mean(axis=0).astype(int)

    # Crop the image 50 pixels from the centroid in x and y direction
    cropped_mri_slice_masked = masked_mri_data[
        max(centroid_y - crop_size, 0):min(centroid_y + crop_size, mri_data.shape[0]),
        max(centroid_x - crop_size, 0):min(centroid_x + crop_size, mri_data.shape[1]),
        slice_index
    ]

    cropped_mri_slice_unmasked = mri_data[
        max(centroid_y - crop_size, 0):min(centroid_y + crop_size, mri_data.shape[0]),
        max(centroid_x - crop_size, 0):min(centroid_x + crop_size, mri_data.shape[1]),
        slice_index
    ]

    # Create a new variable for the cropped masked array
    cropped_masked_mri_data = masked_mri_data[
        max(centroid_y - crop_size, 0):min(centroid_y + crop_size, mri_data.shape[0]),
        max(centroid_x - crop_size, 0):min(centroid_x + crop_size, mri_data.shape[1]),
        :
    ]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cropped_mri_slice_unmasked, cmap='gray')
    plt.title(f'Unmasked MRI Slice at Index {slice_index}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_mri_slice_masked, cmap='gray')
    plt.title(f'Masked MRI Slice at Index {slice_index}')
    plt.axis('off')

    plt.show()

    return cropped_masked_mri_data

# Initialize a list to store masked MRI data for all patients
all_masked_mri_data_list = []

# Loop over all patients and store the masked MRI data
for patient_index in range(len(df_patients)):
    print(f"Processing patient {patient_index + 1}/{len(df_patients)}")
    masked_mri_data = process_patient_data(patient_index, slice_index=25, crop_size=60)
    all_masked_mri_data_list.append(masked_mri_data)

# Convert the list to a numpy array
all_masked_mri_data_array = np.array(all_masked_mri_data_list)

# Print the shape of the resulting array
print("Shape of all masked MRI data array:", all_masked_mri_data_array.shape)

# Ensure the length of the phenotype column matches the number of patients
assert len(phenotype_column) == len(all_masked_mri_data_list), "Mismatch in number of patients and phenotype data."

# Convert the list to a numpy array
all_masked_mri_data_array = np.array(all_masked_mri_data_list)

# Create the X pairs
X= all_masked_mri_data_array

# Print the shapes of X and Y to verify
print("Shape of all masked MRI data array (X):", all_masked_mri_data_array.shape)

# Save the X and Y numpy arrays to disk
np.save('X.npy', X)

#In case the data needs to be plotted
plt.subplot()
plt.imshow(X[5,:,:,0], cmap='gray')
plt.axis('off')
plt.show()

'''
###############################################
###############################################
#IF ALREADY X and Y EXIST, RUN CNN v

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.video import r3d_18
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load the X and Y numpy arrays from disk
# Extract the phenotype column (second column) from the Excel file
excel_path = 'D:/ThighData.xlsx'
df_patients = pd.read_excel(excel_path)
phenotype_column = df_patients.iloc[:, 1].values

X = np.load('X.npy', allow_pickle=True)
Y = phenotype_column
print(X.shape)
print(Y)

# Reshape the X array to include only slices 6 to 35 (30 slices)
X = X[:, :, :, 6:36]

# Normalize the data
scaler = StandardScaler()
X_shape = X.shape
X = X.reshape(-1, X_shape[-1])
X = scaler.fit_transform(X)
X = X.reshape(X_shape)

# Encode the labels (Y) to integers
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Convert labels to categorical one-hot encoding using PyTorch
Y_categorical = np.eye(len(np.unique(Y_encoded)))[Y_encoded]

# Define cross-validation strategy
kf = StratifiedKFold(n_splits=5)

# Initialize lists to store results
all_fpr = []
all_tpr = []
all_roc_auc = []

for train_index, test_index in kf.split(X, Y_encoded):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y_categorical[train_index], Y_categorical[test_index]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    # Create DataLoader for training and test sets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load the pretrained 3D ResNet model
    model = r3d_18(pretrained=True)

    # Modify the first convolutional layer to accept 1-channel input
    model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, Y_categorical.shape[1])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluate the model
    model.eval()
    Y_pred_prob = []
    Y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            Y_pred_prob.append(outputs.numpy())
            Y_true.append(labels.numpy())

    Y_pred_prob = np.concatenate(Y_pred_prob, axis=0)
    Y_true = np.concatenate(Y_true, axis=0)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(Y_categorical.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)

# Plot ROC curves for each class
plt.figure()
for i in range(Y_categorical.shape[1]):
    mean_fpr = np.mean([fpr[i] for fpr in all_fpr], axis=0)
    mean_tpr = np.mean([tpr[i] for tpr in all_tpr], axis=0)
    mean_roc_auc = np.mean([roc_auc[i] for roc_auc in all_roc_auc])
    plt.plot(mean_fpr, mean_tpr, label=f'Class {label_encoder.inverse_transform([i])[0]} (area = {mean_roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()
