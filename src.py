# ============================================================== IMPORTS ============================================================= #
### DATA HANDLING ###
import pandas as pd
import numpy as np

### MODEL BUILDING AND TRAINING ###
import torch
import torch.nn as nn                         # NN LAYERS
import torch.optim as optim                   # OPTIMISATION
import torch.nn.functional as F               # ACTIVATION FUNCTIONS
from torch.utils.data import DataLoader       # CREATE DATA LOADERS
from torchvision import datasets, transforms  # DATASET UTILITIES AND TRANSFORMATION

### DATASET SPLITTING ###
from sklearn.model_selection import train_test_split  # TRAINING AND TEST SETS

### DATA VISUALISATION - PLOTTING ###
import matplotlib.pyplot as plt

### MISC
import time as t  # TIME TRACKING
# ==================================================================================================================================== #



# ========================================================= DATASET OBJECT =========================================================== #
### IMAGE DIMENSIONS FOR RESIZING ###
target_height = 512
target_width = 512

### DIRECTORY WHERE THE DATA IS STORED ###
dataset_dir = "Pizza_Or_Not_Pizza_Dataset"

### TRANSFORMATION: Resizes images to 512x512 and converts to Tensor, then Normalizes ###
transform = transforms.Compose([
    transforms.Resize((target_height, target_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# DEBUG LINE #
print("Creating Dataset... Please wait.")
### CREATES DATASET FROM SPECIFIED FOLDER ###
dataset = datasets.ImageFolder(dataset_dir, transform=transform)
# DEBUG LINE #
print("Dataset creation finished!\n")

# ==================== SPLITTING DATASET INTO TRAINING SET & TEST SET ==================== #

### SEED ###
seed = 777777 # INITIAL VALUE 777777
torch.manual_seed(seed)
# DEBUG LINE #
print("Seed Used:", seed)

# DEBUG LINE #
print("Splitting Dataset... Please wait. \n")
### SPLITS DATASET: Training and Test-Sets (80% Train, 20% Test) ###
train_set, test_set = train_test_split(dataset, shuffle = True, test_size = 0.2)
# DEBUG LINE #
print("Dataset split finished!\n")
# ==================================================================================================================================== #



# ========================================================= HYPERPARAMETERS ========================================================== #
total_epochs = 100          # KEEP - initial value 10.
learning_rate = 0.000005    # KEEP - initial value 0.001.
specified_momentum = 0.4    # KEEP - initial value 0.9.
dropout_rate = 0.35         # KEEP - initial value 0.5.
# ==================================================================================================================================== #




# =========================================================== DATA LOADERS =========================================================== #
### BATCH SIZE: For Training and Testing ###
set_batch_size = 32 # KEEP.
# DEBUG LINE #
print("Batch size:", set_batch_size)
print("\n""Creating Data Loaders... Please wait.")

### TRAINING DATA LOADER ###
train_loader = DataLoader(train_set, batch_size = set_batch_size, shuffle = True)
### TESTING DATA LOADER ###
test_loader = DataLoader(test_set, batch_size = set_batch_size, shuffle = False)

# DEBUG LINE #
print("Finished!\n\n\nPREPARATION FINISHED\n\n\n")
print("------ HYPERPARAMETERS ------")
print("Epochs:", total_epochs)
print("Learning Rate:", learning_rate)
print("Momentum:", specified_momentum)
print("Drop-Out Rate:", dropout_rate)
print("\n")
# =================================================================================================================================== #



# ====================================================== NEURAL NETWORK CLASS ======================================================= #
class NeuralNetwork(nn.Module):
    def __init__(self, _dropout_rate):
        super().__init__()
        
        ### CONVOLUTIONAL LAYERS ###
        self.conv1 = nn.Conv2d(3, 32, 3, 1) # FIRST LAYER
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # SECOND LAYER
        
        ### POOLING LAYERS ###
        self.pool = nn.MaxPool2d(2, 2) # MAX POOLING
        
        ### FLATTEN LAYERS ###
        self.flatten = nn.Flatten() # Flattens Tensor for Fully Connected layers
        
        ### FULLY CONNECTED LAYERS ###
        self.fc1 = nn.Linear(64 * 126 * 126, 128) # FIRST LAYER
        self.fc2 = nn.Linear(128, 2) # SECOND LAYER - OUTPUT LAYER
        
        ### DROPOUT LAYER - REGULARISATION ###
        self.dropout = nn.Dropout(_dropout_rate)

    
    ### FORWARDS PROPAGATION THROUGH LAYERS ###
    def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)

            x = F.relu(self.conv2(x))
            x = self.pool(x)
            
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x
# =================================================================================================================================== #



# ===================================================== TRAINING NEURAL NETWORK ===================================================== #
### DEVICE SELECTION: Selects the appropriate device for Training: GPU, MPS (Apple Silicon) or CPU ###
# UNCOMMENT ON NEED #

# Use GPU if available
# if torch.cuda.is_available():
#     print("Using Cuda (GPU) for Training\n\n")
#     device = "cuda" 
# elif torch.backends.mps.is_available():
#     print("Using MPS for Training\n\n")
#     device = "mps"
# else:
#     print("Using CPU for Training\n\n")
#     device = "cpu"

### FORCES USE OF MPS - (Metal Performance Shaders) ###
device = torch.device("mps")


### INSTANCE OF NEURAL NETWORK & TRANSFER TO DEVICE ###
c_neural_network = NeuralNetwork(dropout_rate).to(device)

### SETS OPTIMISER ###
optimiser = optim.SGD(c_neural_network.parameters(), lr = learning_rate, momentum = specified_momentum)

### LEARNING RATE SCHEDULER ###
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode = 'min', factor = 0.1, patience = 10, verbose = True)

### LOSS FUNCTION ###
loss_func = nn.CrossEntropyLoss() # INCLUDES SOFTMAX

### ARRAYS STORING LOSS & ACCURACY VALUES: for later plotting ###
training_losses = []
validation_losses = []
test_accuracies = []

### EARLY STOPPING CRITERIA ###
best_val_loss = float('inf')
patience = 10  # Number of epochs to wait for improvement before stopping.
wait = 0
min_delta = 0.0005  # Minimum change to qualify for improvement.
# ================================================================================================================================== #



# ========================================================== TRAINING LOOP ========================================================= #
print("Used device:", device, "\n")
print("Training Neural Network... Please wait.")
start_time = t.time() # START TIMER

### TRAINING LOOP ###
for epoch in range(total_epochs):
    c_neural_network.train() # Sets the network to training mode.
    total_training_loss = 0.0 # Total training loss for the epoch.
    
    ### TRAINING PROCESS ###
    for batch, pair in enumerate(train_loader):
        images, labels = pair
        images = images.to(device)
        labels = labels.to(device)
        
        optimiser.zero_grad() # Zeros gradients for accumulation prevention.

        label_predictions = c_neural_network(images) # FORWARD PASS
        loss = loss_func(label_predictions, labels) # COMPUTES LOSS
        
        loss.backward() # Backward pass for gradient computation
        optimiser.step() # Updating model parameters
        
        total_training_loss += loss.item()
        
    average_training_loss = total_training_loss / len(train_loader) # Computes average Training Loss per Epoch.
    training_losses.append(average_training_loss) # Append Training Loss to array.


    # ==================== Calculating Accuracy & Validation Loss ==================== #
    c_neural_network.eval()
    correct_predictions = 0
    total_predictions = 0
    total_validation_loss = 0.0
    
    ### STOPS GRADIENT COMPUTATION ###
    with torch.no_grad():
        for batch, pair in enumerate(test_loader):
            images, labels = pair
            images = images.to(device)
            labels = labels.to(device)

            ### LABEL PREDICTIONS ###
            label_predictions = c_neural_network(images)
            ### VALIDATION LOSS ###
            validation_loss = loss_func(label_predictions, labels)

            ### OBTAINS PREDICTED LABELS ###
            _, predicted = torch.max(label_predictions.data, 1)
            ### ADDS BATCH SIZE TO total_predictions ###
            total_predictions += labels.size(0)

            ### CHECKS HOW MANY PREDICTIONS WERE CORRECT ###
            correct_predictions += (predicted == labels).sum().item()
            ### ADDS ALL VALIDATION LOSS FOR THE EPOCH ###
            total_validation_loss += validation_loss.item()
            
    average_validation_loss = total_validation_loss / len(test_loader) # COMPUTES AVERAGE VALIDATION LOSS FOR EPOCHS
    validation_losses.append(average_validation_loss) # APPEND VALIDATION LOSS TO ARRAY
    scheduler.step(average_validation_loss) # UPDATE LEARNING RATE
    
    accuracy = correct_predictions / total_predictions # COMPUTES ACCURACY
    test_accuracies.append(accuracy) # ADDS ACCURACY TO ARRAY

    ### EARLY STOP CHECK ###
    if average_validation_loss < best_val_loss - min_delta:
        best_val_loss = average_validation_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Stopped early at epoch {epoch + 1}")
            break
            
    # DEBUG LINE: displays Loss & Accuracy #
    print ("\n---------- EPOCH:", epoch + 1, "/", total_epochs, "----------")
    print("Training Loss:", average_training_loss)
    print("Validation Loss:", average_validation_loss)
    print("Accuracy:", accuracy)
    
print("\nTraining Complete!")

### TIMER ###
end_time = t.time()
total_time = end_time - start_time
mins, sec = divmod(total_time, 60)
print("\nTime Elapsed: ", round(mins), "Mins", round(sec), "Seconds") # DISPLAYS ELAPSED TIME
# ==================================================================================================================================== #



# ==================================================== PLOT LOSS & ACCURACY GRAPHS =================================================== #
### PLOTTING ACCURACY AND LOSS OVER EPOCH ###
plt.figure(figsize = (12, 8))  # FIGURE SIZE

### PLOTTING TRAINING AND VALIDATION LOSS ###
plt.subplot(2, 1, 1)  # 2 Rows, 1 Column, 1st Subplot.
#### X-AXIS: Uses actual length of training_losses ###
plt.plot(range(1, len(training_losses) + 1), training_losses, label = "Training Loss")
plt.plot(range(1, len(validation_losses) + 1), validation_losses, label = "Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss Over Epochs")
plt.legend()
plt.tight_layout()

### PLOTTING TEST ACCURACY ###
plt.subplot(2, 1, 2)  # 2 Rows, 1 Column, 2nd Subplot.
### X-AXIS: Uses actual length of test_accuracies ###
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label = "Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy Over Epochs")
plt.legend()
plt.tight_layout()

### SAVES PLOTS: Hyperparameters plugged in the filename ###
plot_name = f"Loss & Accuracy Plot (lr={learning_rate}, momentum={specified_momentum}, dropout={dropout_rate})"
plt.savefig(f"Plots/{plot_name}.png")
plt.show()

print("\nPlot Saved")
# ==================================================================================================================================== #