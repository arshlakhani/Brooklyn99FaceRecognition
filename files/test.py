'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_3023
# Author List:		Arsh Lakhani, Gurkirat Singh
# Filename:			task_1a.py
# Functions:	    [ideantify_features_and_targets, load_as_tensors,
# 					 model_loss_function, model_optimizer, model_number_of_epochs, training_function,
# 					 validation_functions ]

####################### IMPORT MODULES #######################
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

file = pd.read_csv(r'Task_1A\task_1a_dataset.csv')


# print(file.head())


##############################################################

def data_preprocessing(task_1a_dataframe):
    '''
    Purpose:
    ---
    This function will be used to load your csv dataset and preprocess it.
    Preprocessing involves cleaning the dataset by removing unwanted features,
    decision about what needs to be done with missing values etc. Note that
    there are features in the csv file whose values are textual (eg: Industry,
    Education Level etc)These features might be required for training the model
    but can not be given directly as strings for training. Hence this function
    should return encoded dataframe in which all the textual features are
    numerically labeled.

    Input Arguments:
    ---
    task_1a_dataframe: [Dataframe]
                          Pandas dataframe read from the provided dataset

    Returns:
    ---
    encoded_dataframe : [ Dataframe ]
                          Pandas dataframe that has all the features mapped to
                          numbers starting from zero

    Example call:
    ---
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    '''

    #################	ADD YOUR CODE HERE	##################

    ##########################################################
    for col in ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain']:
        task_1a_dataframe[col].fillna(task_1a_dataframe[col].mean(), inplace=True)
    # fillna fills missing values with mean of the column
    # inplace =True updates the dataframe directly without creatiing a new one
    # .mean() calculates the mean of the column to fill missing values

    for col in ['Education', 'City', 'Gender', 'EverBenched']:
        task_1a_dataframe[col].fillna(task_1a_dataframe[col].mode()[0], inplace=True)
    # fillna fills missing values with mode of the column
    # .mode()[0] returns the most frequent value in the column to fill missing values
    # its returning multiple mode so we select first index one
    label_encoder = LabelEncoder()
    for col in ['Education', 'City', 'Gender', 'EverBenched']:
        task_1a_dataframe[col] = label_encoder.fit_transform(task_1a_dataframe[col])
    # it is label encoding the categorical columns
    # it convert string to numeric values

    encoded_dataframe = task_1a_dataframe

    return encoded_dataframe


def identify_features_and_targets(encoded_dataframe):
    '''
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and second
    item is the target label

    Input Arguments:
    ---
    encoded_dataframe : [ Dataframe ]
                        Pandas dataframe that has all the features mapped to
                        numbers starting from zero

    Returns:
    ---
    features_and_targets : [ list ]
                            python list in which the first item is the
                            selected features and second item is the target label

    Example call:
    ---
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    '''

    #################	ADD YOUR CODE HERE	##################

    ##########################################################
    targetData = encoded_dataframe['LeaveOrNot']
    featuresData = encoded_dataframe.drop(['LeaveOrNot'], axis=1)
    features_and_targets = [featuresData, targetData]

    return features_and_targets


def load_as_tensors(features_and_targets):
    '''
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training
    and validation, and then load them as as tensors.
    Training of the model requires iterating over the training tensors.
    Hence the training sensors need to be converted to iterable dataset
    object.

    Input Arguments:
    ---
    features_and targets : [ list ]
                            python list in which the first item is the
                            selected features and second item is the target label

    Returns:
    ---
    tensors_and_iterable_training_data : [ list ]
                                            Items:
                                            [0]: X_train_tensor: Training features loaded into Pytorch array
                                            [1]: X_test_tensor: Feature tensors in validation data
                                            [2]: y_train_tensor: Training labels as Pytorch tensor
                                            [3]: y_test_tensor: Target labels as tensor in validation data
                                            [4]: Iterable dataset object and iterating over it in
                                                 batches, which are then fed into the model for processing

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''

    #################	ADD YOUR CODE HERE	##################
    feature, target = features_and_targets
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)
    # split data into train and test with 80-20 ratio
    X_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=15, shuffle=True)
    tensors_and_iterable_training_data = [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader]

    ##########################################################

    return tensors_and_iterable_training_data


class Salary_Predictor(nn.Module):
    '''
    Purpose:
    ---
    The architecture and behavior of your neural network model will be
    defined within this class that inherits from nn.Module. Here you
    also need to specify how the input data is processed through the layers.
    It defines the sequence of operations that transform the input data into
    the predicted output. When an instance of this class is created and data
    is passed through it, the forward method is automatically called, and
    the output is the prediction of the model based on the input data.

    Returns:
    ---
    predicted_output : Predicted output for the given input data
    '''

    def _init_(self):
        super(Salary_Predictor, self)._init_()
        '''
        Define the type and number of layers
        '''
        #######	ADD YOUR CODE HERE	#######
        self.v1 = nn.Linear(8, 32)
        # layer with 32 node
        self.v2 = nn.Linear(32, 64)
        # laer with 64 nodes
        self.v3 = nn.Linear(64, 1)

    ###################################

    def forward(self, x):
        '''
        Define the activation functions
        '''
        #######	ADD YOUR CODE HERE	#######
        predicted_output = F.relu(self.v1(x))  # relu give 0 or positive value which is high
        # Apply ReLU activation after first linear layer
        predicted_output = F.relu(self.v2(predicted_output))
        predicted_output = self.v3(predicted_output)

        ###################################

        return predicted_output


def model_loss_function():
    '''
    Purpose:
    ---
    To define the loss function for the model. Loss function measures
    how well the predictions of a model match the actual target values
    in training data.

    Input Arguments:
    ---
    None

    Returns:
    ---
    loss_function: This can be a pre-defined loss function in PyTorch
                    or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    '''
    #################	ADD YOUR CODE HERE	##################
    loss_function = nn.BCEWithLogitsLoss()  # apply Binary Cross Entropy loss for binary classification
    # sigmoid is appl in aboce only
    ##########################################################

    return loss_function


def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model. Optimizer is responsible
    for updating the parameters (weights and biases) in a way that
    minimizes the loss function.

    Input Arguments:
    ---
    model: An object of the 'Salary_Predictor' class

    Returns:
    ---
    optimizer: Pre-defined optimizer from Pytorch

    Example call:
    ---
    optimizer = model_optimizer(model)
    '''
    #################	ADD YOUR CODE HERE	##################

    ##########################################################
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    return optimizer


def model_number_of_epochs():
    '''
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    number_of_epochs: [integer value]

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    '''
    #################	ADD YOUR CODE HERE	##################

    ##########################################################
    no_of_epochs = 50
    return no_of_epochs


def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. model: An object of the 'Salary_Predictor' class
    2. number_of_epochs: For training the model
    3. tensors_and_iterable_training_data: list containing training and validation data tensors
                                             and iterable dataset object of training tensors
    4. loss_function: Loss function defined for the model
    5. optimizer: Optimizer defined for the model

    Returns:
    ---
    trained_model

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

    '''
    #################	ADD YOUR CODE HERE	##################
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader = tensors_and_iterable_training_data

    for epoch in range(number_of_epochs):
        for no_use, (data, target) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Compute the loss
            loss = loss_function(output, target)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        print(f"Epoch {epoch} completed")

    ##########################################################

    return model


def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilise the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. trained_model: Returned from the training function
    2. tensors_and_iterable_training_data: list containing training and validation data tensors
                                             and iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

    '''
    #################	ADD YOUR CODE HERE	##################
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader = tensors_and_iterable_training_data
    trained_model.eval()
    with torch.no_grad():
        output = trained_model(X_test_tensor)
    pred = torch.round(torch.sigmoid(output))
    model_accuracy = accuracy_score(y_test_tensor, pred)

    ##########################################################

    return model_accuracy


########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "_main_":
    # reading the provided dataset csv file using pandas library and
    # converting it to a pandas Dataframe
    task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    # print(features_and_targets[0].head)
    # print(features_and_targets[1].head)

    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

    # model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor()

    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    # training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data,
                                      loss_function, optimizer)

    # validating and obtaining accuracy
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "dustbin.pth")  # save it