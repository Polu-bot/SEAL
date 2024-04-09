import pyautogui
import shutil
import os
import pathlib
import glob
import keyboard
import datetime
import torch
import gpytorch
import numpy as np
import random
from osgeo import gdal
import sklearn.metrics
import csv
import pickle
from numpy.linalg import inv
import matplotlib.pyplot as plt
import keyboard
import scipy.stats as ss
import pandas as pd
import re
import math
import time
import subprocess

recordDataScript = "../scripts/usb_4_mic_array-master/acquire_data.py"
processDataScript = "../scripts/usb_4_mic_array-master/record_data.py"

PROCESSASD = True
CORRELATIONCODE = True
MOVEFILES = True

FolderPath = r"C:\Users\shuai\Desktop\ASD_Data\RoSE Lab Project\\"
TrialName = "S_SEAL_Trial_1"

grid_length = 11
grid_width = 11

SNAKEPATTERN = False
SPIRALPATTERN = False

kernel_type = 'RBF'
poly_rank = 4
r_disp = 6.0
constraint_flag = 1
local_flag = 1

if local_flag ==1:
    explore_name = 'local'
elif local_flag == 2:
    explore_name = 'NN'
else:
    explore_name = 'global'
    
if constraint_flag == 0:
    con_name = 'unconstrained'
else:
    con_name = 'constrained'

fig = plt.figure()


# Function Definitions ##################################################################################

# calculate RKHS norm from covariance
def RKHS_norm(y,sigma,K):
    n_row, n_col = K.shape
    alpha = inv(K + sigma**2 * np.eye(n_row)) @ y
    return alpha.transpose() @ K @ alpha

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 2))
        #self.covar_module.base_kernel.lengthscale = length_scale
        if kernel_type == 'RBF':
        # RBF Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 2)) 
        elif kernel_type == 'Matern':
        # Matern Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims = 2, nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def kernel_print(i, training_iter, loss, model):
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.detach().numpy()[0][0],
            model.covar_module.base_kernel.lengthscale.detach().numpy()[0][1],
            model.likelihood.noise.detach().numpy()
        ))
    
def GPtrain(x_train, y_train, training_iter):
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # train GP model
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        kernel_print(i, training_iter, loss, model)
        optimizer.step()
    
    return likelihood, model, optimizer, output, loss

def GPEval(x_test, model, likelihood):
    model.eval()
    likelihood.eval()
    
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_test))
        f_preds = model(x_test)
        f_mean = f_preds.mean
        f_var = f_preds.variance
        #f_var = np.diag(f_preds.lazy_covariance_matrix.numpy())
        f_covar = f_preds.covariance_matrix
    
    with torch.no_grad():
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

    # print('mean', f_mean)
    # print('var', f_var)
        
    return observed_pred, lower, upper, f_var, f_covar, f_mean

def unique_sample(i_sample,i_set,i_train,i_max,x):
    if i_sample <= i_max and i_sample >= 0:
        if i_sample not in i_train:
            i_new = i_sample
        else:
            i_set_unique = set(i_set)-set(i_train)
            if not i_set_unique:
                return None
            i_set_unique = list(i_set_unique)
            x_start = x[i_sample,:]
            x_disp = np.sqrt((x[i_set_unique,0]-x_start[0])**2 + (x[i_set_unique,1]-x_start[1])**2)
            # disp_i = np.abs(np.array(i_set_unique)-np.array(i_sample))
            i_new =i_set_unique[np.argmin(x_disp)]
    elif i_sample > i_max:
        i_new = unique_sample(i_sample-1,i_set,i_train,i_max)
    else:
        i_new = unique_sample(i_sample+1,i_set,i_train,i_max)
    return i_new

def create_folder_with_suffix(folder_path):
    i = 1
    while True:
        new_folder_path = folder_path + f"_{i}" if i > 1 else folder_path
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            return new_folder_path
        i += 1

new_folder_path = create_folder_with_suffix(FolderPath + TrialName)

def move_txt_files(source_folder, destination_folder):
    # Ensure destination folder exists, create if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through files in source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt") or filename.endswith(".asd"):
            # Build paths for source and destination files
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            # Move the file
            shutil.move(source_file, destination_file)
            print(f"Moved {filename} to {destination_folder}")


def snake_pattern(grid):
    result = []
    for i, row in enumerate(grid):
        if i % 2 == 0:
            result.extend(row)
        else:
            result.extend(reversed(row))
    return result

def spiral_traversal(grid):
    result = []
    while grid:
        result.extend(grid.pop(0))  # Traverse top row
        if grid and grid[0]:
            for row in grid:
                result.append(row.pop())  # Traverse right column
        if grid:
            result.extend(grid.pop()[::-1])  # Traverse bottom row in reverse
        if grid and grid[0]:
            for row in reversed(grid):
                result.append(row.pop(0))  # Traverse left column
    return result

def snake_pattern(grid):
    result = []
    for i, row in enumerate(grid):
        if i % 2 == 0:
            result.extend(row)
        else:
            result.extend(reversed(row))
    return result

def update_visualization(next_point, sampled_points):
    sampled_points = sampled_points[:-1]
    grid = np.zeros((grid_length, grid_width)) 
    plt.clf() 
    plt.imshow(grid, cmap='gray', extent=[-0.5, grid_width - 0.5, -0.5, grid_length - 0.5]) 
    plt.plot(np.array(next_point)[0], np.array(next_point)[1], marker='o', color='green') 
    plt.scatter(np.array(sampled_points)[:, 0], np.array(sampled_points)[:, 1], marker='x', color='red') 
    plt.title('Rover Exploration')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(np.arange(0, grid_width, 1))
    plt.yticks(np.arange(0, grid_length, 1))
    plt.grid(True)
    plt.pause(0.5)  # Pause to allow the plot to be displayed
    image_file_path = os.path.join(new_folder_path, "GPAL_Path.png")
    plt.savefig(image_file_path)

def sample_disp_con(x,x_start,r_disp):
    # x_start = x[i_start,:]
    x_disp = np.sqrt((x[:,0]-x_start[0])**2 + (x[:,1]-x_start[1])**2)# + (x[:,2]-x_start[2])**2)
    i_con = np.argwhere(x_disp<=r_disp)
    i_con = np.sort(i_con)
    return list(i_con[:,0])

def process_data():
    # this should give us x coordinates, y coordinates, SPL level and angle
    subprocess.run(["python", recordDataScript])
    subprocess.run(["sudo python", processDataScript])

# Define Test Space ###################################################################
x_coordinates = np.arange(grid_length)
y_coordinates = np.arange(grid_width)
x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
x_true = np.stack((x_grid, y_grid), axis=-1)
x_true = x_true.tolist()
x_true = [item for sublist in x_true for item in sublist]
x_true = np.array(x_true)

n = int(np.sqrt(len(x_true)))
m = n
grid = x_true.reshape((n, m, 2))

num = 0
coordinate_to_number = {}
for i, row in enumerate(grid):
    for j, coord in enumerate(row):
        coordinate_to_number[tuple(coord)] = num
        num += 1

# Data normalizaton/standardization
x_center_all = np.mean(x_true,0)
x_disp = np.sqrt((x_true[:,0]-x_center_all[0])**2 + (x_true[:,1]-x_center_all[1])**20)
i_min = np.argmin(x_disp)
x_center = x_true[i_min,:]
x_true = x_true - x_center


# Training Parameters ######################################################################
n_samples = len(x_true)
r_NN = np.sqrt(2)#-0.29 #np.sqrt(3)*0.25 
r_con = r_NN
i_seq = list(range(0,n_samples))


if SNAKEPATTERN or SPIRALPATTERN:
    sample_iter = grid_length * grid_width
else:
    sample_iter = int(n_samples/2)
training_iter = 100
i_train = []
var_iter = []
var_iter_local = []
var_iter_global = []
lengthscale = []

covar_global = []
covar_trace = []
covar_totelements = []
covar_nonzeroelements = []
f2_H_global = []
f2_H_local = []
flags = []

noise = []

# Random Walk ################################################################################
if SNAKEPATTERN:
    gridIndex = [[i + j * grid_length for i in range(grid_width)] for j in range(grid_length)]
    i_train_known = snake_pattern(gridIndex)
    i_train.append(i_train_known[0])
    i_sample = i_train_known[0]
elif SPIRALPATTERN:
    gridIndex = [[i + j * grid_length for i in range(grid_width)] for j in range(grid_length)]
    i_train_known = spiral_traversal(gridIndex)
    i_train.append(i_train_known[0])
    i_sample = i_train_known[0]
else:
    random.seed(42)
    i_0 = random.randrange(n_samples) #%% randomly initialize location
    i_train.append(i_0)
    for i in range(1):
        i_sample_set = sample_disp_con(x_true,x_true[i_train[-1]],r_NN) # nearest neighbor (within 0.25 km)
        #i_sample_set = sample_disp_con(x_true,x_true[i_train[-1]],r_con) # within 1 km
        i_sample = i_sample_set[random.randrange(len(i_sample_set))]
        i_train.append(int(i_sample))
    i_train = list(set(i_train))
    print("Number of initial points:",len(i_train))

#Creating first 10 payload input values corresponsing to above
y_obs = np.array([])
testedWaypoints = []
count = 0
iteration = 0

for coordinate in x_true[i_train,:]:
    coordinatePoint = coordinate + x_center
    print(coordinatePoint)
    testedWaypoints.append(coordinatePoint)
    
    coord = tuple(coordinatePoint.flatten())
    if coord in coordinate_to_number:
        flag = coordinate_to_number[coord]

    print("Rover Location: ",flag, coord)
    if count != 0:
        rover_path = [item.tolist() if isinstance(item, np.ndarray) else item for item in testedWaypoints]
        update_visualization(list(coordinatePoint), rover_path)
   
    cont = True
    while cont:
        print("Collect acoustic data")
        keyboard.wait('tab')
        cont = process_data()

    
    csv_file_path = new_folder_path+"\AcousticData.csv"
    df = pd.read_csv(csv_file_path)
    acousticVal = df.iloc[iteration, 1]
    #user_input = get_valid_float_input("Please enter payload input: ")
    print(acousticVal)
    y_obs = np.append(y_obs, float(acousticVal))
    file_path = os.path.join(new_folder_path, "GPAL_tested_coordinates.txt")
    testedCoord = open(file_path, "a")
    testedCoord.write(str(coordinatePoint) + '\n')
    testedCoord.close()
    file_path = os.path.join(new_folder_path, "GPAL_acoustic_values.txt")
    acouVals = open(file_path, "a")
    acouVals.write(str(acousticVal) + '\n')
    acouVals.close()
    count+=1
    iteration += 1

# EXPLORATION PHASE ###################################################################

for j in range(sample_iter):
    # define training data
    x_train = torch.from_numpy(x_true[i_train,:])
    y_train = torch.from_numpy(y_obs)

    x_train = x_train.float()
    y_train = y_train.float()

    likelihood, model, optimizer, output, loss = GPtrain(x_train, y_train, training_iter)
    noise.append(model.likelihood.noise.detach().numpy())

    # Test points are regularly spaced centered along the last index bounded by index displacement
    i_con = sample_disp_con(x_true,x_true[i_train[-1]],r_con)
    x_test_local = torch.from_numpy(x_true[i_con,:]) # x_test is constrained to motion displacement
    x_test_global = torch.from_numpy(x_true[i_seq,:]) # x_test is the entire dataset
    
    x_test_local = x_test_local.float()
    x_test_global = x_test_global.float()
    
    # Evaluate RMS for local 
    observed_pred_local, lower_local, upper_local = GPEval(x_test_local, model, likelihood)
    with torch.no_grad():
        f_preds = model(x_test_local)
        y_preds = likelihood(model(x_test_local))
        f_mean = f_preds.mean
        f_var_local = f_preds.variance
        f_covar = f_preds.covariance_matrix
    var_iter_local.append(max(f_var_local.numpy()))  

    # and global
    observed_pred_global, lower_global, upper_global = GPEval(x_test_global, model, likelihood)
    with torch.no_grad():
        f_preds = model(x_test_global)
        y_preds = likelihood(model(x_test_global))
        f_mean = f_preds.mean
        f_var_global = f_preds.variance
        f_covar = f_preds.covariance_matrix
    var_iter_global.append(max(f_var_global.numpy()))

    # evaluate covariance properties
    covar_global.append(f_covar)
    covar_trace.append(np.trace(f_covar.detach().numpy()))
    covar_totelements.append(np.size(f_covar.detach().numpy()))
    covar_nonzeroelements.append(np.count_nonzero(f_covar.detach().numpy()))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    sigma = 0.000289305 
    # and finally evaluate RKHS norm
    K_global = output._covar.detach().numpy()
    y_global = y_train.numpy().reshape(len(y_train),1)
    f2_H_sample = RKHS_norm(y_global,sigma,K_global)
    f2_H_global.append(f2_H_sample[0,0])
    
    n_set = len(i_train)
    n_sub = math.floor(n_set/2)
    i_sub = random.sample(range(1,n_set),n_sub)
    i_sub.sort()
    K_local = K_global[np.ix_(i_sub,i_sub)]
    y_local = y_global[i_sub]
    f2_H_sample = RKHS_norm(y_local,sigma,K_local)
    if j == 1 and SNAKEPATTERN and SPIRALPATTERN:
        f2_H_local.append(f2_H_sample[0,0])
    # pick the next point to sample by maximizing local variance and minimizing distance by sampling nearest neighbor along the way
    try: 
        if SNAKEPATTERN or SPIRALPATTERN:
            # waypoint within r_con with maximum variance, nearest neighbor along the way
            uncertainty = upper_local-lower_local
            i_max = np.argmax(uncertainty)
            x_max = x_test_local[i_max,:].numpy()
            i_NN = sample_disp_con(x_true,x_true[i_train[-1]],r_NN)
            dx_NN = np.sqrt((x_true[i_NN,0]-x_max[0])**2 + (x_true[i_NN,1]-x_max[1])**2)
            i_dx = np.argsort(dx_NN)
            i_train.append(i_train_known[j+1])
            i_sample = i_train_known[j+1]
        elif local_flag ==1 or local_flag ==2:
            # waypoint within r_con with maximum variance, nearest neighbor along the way
            uncertainty = upper_local-lower_local
            i_max = np.argmax(uncertainty)
            x_max = x_test_local[i_max,:].numpy()
            i_NN = sample_disp_con(x_true,x_true[i_train[-1]],r_NN)
            dx_NN = np.sqrt((x_true[i_NN,0]-x_max[0])**2 + (x_true[i_NN,1]-x_max[1])**2)
            i_dx = np.argsort(dx_NN)
            i_sample = []
            j = 0
            while not np.array(i_sample).size:
                i_sample = unique_sample(i_NN[i_dx[j]],i_con,i_train,n_samples-1,x_true)
                count1 = 0
                while i_sample is None:
                    count1+=1
                    i_sample_set = sample_disp_con(x_true,x_true[i_train[-1]],np.sqrt(count1)-0.29)
                    i_sample = i_sample_set[random.randrange(len(i_sample_set))]
                    if i_sample in i_train:
                        i_sample = None
                    print("OHNO: NONUNIQUEPOINT")
                j = j+1
            i_train.append(int(i_sample))
        else:
            # waypoint within global space with maximum variance, directly go
            uncertainty = upper_global-lower_global
            i_max = np.argmax(uncertainty)
            # waypoint within entire space with max variance, nearest neighbor
            if constraint_flag == 1:
                a = x_test_global[i_max,:].numpy()
                i_NN = sample_disp_con(x_true,x_true[i_train[-1]],r_NN)
                dx_NN = np.sqrt((x_true[i_NN,0]-x_max[0])**2 + (x_true[i_NN,1]-x_max[1])**2)
                i_dx = np.argsort(dx_NN)
                # finds the nearest neigbor along the way to the point of highest variance
                i_sample = []
                j = 0
                while not np.array(i_sample).size:
                    i_sample = unique_sample(i_NN[i_dx[j]],i_con,i_train,n_samples-1,x_true)
                    j = j+1
                i_train.append(int(i_sample))
            else:
                i_train.append(int(i_max))

        coordinatePoint = x_true[i_train[-1]] + x_center
        testedWaypoints.append(list(coordinatePoint))
        coord = tuple(coordinatePoint.flatten())
        if coord in coordinate_to_number:
            flag = coordinate_to_number[coord]

        print("Move rover to: ", flag, coord)
        rover_path = [item.tolist() if isinstance(item, np.ndarray) else item for item in testedWaypoints]
        update_visualization(list(coordinatePoint), rover_path)

        cont = True
        while cont:
            print("Bring up FieldSpecPro and then press tab to process data")
            keyboard.wait('tab')
            cont = process_data()
        
            
        csv_file_path = new_folder_path+"\CorrelationData.csv"
        df = pd.read_csv(csv_file_path)
        correlationVal = df.iloc[iteration, 1]
        #user_input = get_valid_float_input("Please enter payload input: ")
        print(correlationVal)
        y_obs = np.append(y_obs, float(correlationVal))
        file_path = os.path.join(new_folder_path, "GPAL_tested_coordinates.txt")
        testedCoord = open(file_path, "a")
        testedCoord.write(str(coordinatePoint) + '\n')
        testedCoord.close()
        file_path = os.path.join(new_folder_path, "GPAL_correlation_values.txt")
        corrVals = open(file_path, "a")
        corrVals.write(str(correlationVal) + '\n')
        corrVals.close()

        iteration+=1

    except Exception as e:
        print("An exception occurred:", e)
        break


###############################################################################
################################## save data ##################################
###############################################################################
     
class data:
    def __init__(self, x_true, y_obs, i_train, var_iter_global,
                 var_iter_local, x_test_local, x_test_global, covar_global, covar_trace, 
                 covar_totelements, covar_nonzeroelements, f2_H_local, f2_H_global):
        # likelihood, model, optimizer, output, loss,
        self.x_true = x_true
        self.y_obs = y_obs
        self.i_train = i_train
        self.var_iter_global = var_iter_global
        self.var_iter_local = var_iter_local
        self.x_test_local = x_test_local
        self.x_test_global = x_test_global
        self.covar_global = covar_global
        self.covar_trace = covar_trace
        self.covar_totelements = covar_totelements
        self.covar_nonzeroelements = covar_nonzeroelements
        self.f2_H_local = f2_H_local
        self.f2_H_global = f2_H_global

mydata = data(x_true, y_obs, i_train, var_iter_global, 
              var_iter_local, x_test_local, x_test_global,
              covar_global, covar_trace, covar_totelements, covar_nonzeroelements, f2_H_local, f2_H_global)
x_true = mydata.x_true
y_obs = mydata.y_obs
i_train = mydata.i_train
var_iter_global = mydata.var_iter_global
var_iter_local = mydata.var_iter_local
x_test_local = mydata.x_test_local
x_test_global = mydata.x_test_global
covar_global = mydata.covar_global
covar_trace = mydata.covar_trace
covar_totelements = mydata.covar_totelements
covar_nonzeroelements = mydata.covar_nonzeroelements
f2_H_local = mydata.f2_H_local
f2_H_global = mydata.f2_H_global

# Save data to a pickle file
file_path = os.path.join(new_folder_path, "saved_data.pkl")
with open(file_path, 'wb') as file:
    pickle.dump(mydata, file)

move_txt_files(r"C:\Users\shuai\Desktop\ASD_Data\RoSE Lab Project\SpecData", new_folder_path)
 