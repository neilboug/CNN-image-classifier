# 🍕 Pizza or Not Pizza – CNN Image Classifier (PyTorch)

This project implements a **Convolutional Neural Network (CNN)** in **PyTorch** to classify images as either **pizza** or **not pizza**. The model is trained from scratch and includes key techniques such as **dropout**, **learning rate scheduling**, **early stopping**, and **accuracy/loss tracking**.

---

## 🚀 Features

- 🧠 Custom CNN built using **PyTorch**
- 🖼️ Classifies images from a binary dataset (`Pizza` vs `Not Pizza`)
- 🧹 Uses `ImageFolder` for directory-based dataset loading
- 🧪 Train/Test split with `sklearn`
- 🔄 Includes:
  - Convolutional & pooling layers
  - Dropout for regularization
  - Learning rate scheduler
  - Early stopping
- 📈 Saves **accuracy/loss plots** with hyperparameters embedded in the filename

---

## 🗂️ Dataset Instructions

Due to its large size, the dataset is **not included in this repository**.

You must manually download the dataset from Kaggle:

🔗 [Download the dataset here](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza)

After downloading:

1. Extract the folder `Pizza_Or_Not_Pizza_Dataset`.
2. Place it in the root of this project directory like so:

Pizza_Or_Not_Pizza_Dataset/
|── pizza/

| |── image1.jpg

| |── ...

|── not_pizza/

|── image1.jpg

|── ...

---

## 📂 Project Structure

|── src.py # Main training script

|── Pizza_Or_Not_Pizza_Dataset/ # Image dataset (user must download separately)

|── Plots/ # Directory where training plots are saved


---

## ⚙️ How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/pizza-cnn-pytorch.git
cd pizza-cnn-pytorch
```

### 2. Install Dependencies
```bash
Download and extract the dataset as described in the section above. Make sure it sits in the correct folder structure.
```

### 3. Prepare the Dataset
```bash
git clone https://github.com/your-username/pizza-cnn-pytorch.git
cd pizza-cnn-pytorch
```

### 4. Run the Script
```bash
python "src.py"
```

---

## 🔧 Configuration

Adjust training parameters and model setup directly in Pytorch CNN Code.py:

```python
# Image size
target_height = 512
target_width = 512

# Training hyperparameters
total_epochs = 100
learning_rate = 0.000005
specified_momentum = 0.4
dropout_rate = 0.35
set_batch_size = 32
seed = 777777
```
You can also switch devices to CUDA or CPU depending on your setup (currently uses MPS by default).

---

## 📸 Example Output
Plots are saved like:
```mathematica
Plots/Loss & Accuracy Plot (lr=0.000005, momentum=0.4, dropout=0.35).png
```
Here’s a sample output plot:
![Training Plot](Plots/PlotsLoss%20%26%20Accuracy%20Plot%20%28lr%3D0.001%2C%20momentum%3D0.9%2C%20dropout%3D0.5%29.png)







