# MNIST Model with TensorFlow and Keras

This repository contains a complete workflow for training and evaluating an MNIST digit classification model using TensorFlow and Keras. It includes code for model training, evaluation, and inference using uploaded images.

## Repository Contents

- **`mnist_model.keras`**: The saved Keras model file for MNIST digit classification.
- **`MNIST_TSFlow+keras.ipynb`**: Jupyter Notebook for training and saving the MNIST classification model.
- **`ModelExecution.ipynb`**: Jupyter Notebook for loading the saved model and predicting digits from uploaded images.

## Dataset

The dataset used is [MNIST](https://yann.lecun.com/exdb/mnist/) provided by Zalando. It includes the following categories:


![image](https://github.com/user-attachments/assets/0ed027da-a0af-4fea-8f47-2b827b16d88c)

## Model Architecture

The model is a neural network with the following layers:

1. **Flatten Layer**: Converts the 28x28 pixel images into a 1D array.
2. **Dense Layer 1**: 1024 neurons, ReLU activation.
3. **Dense Layer 2**: 256 neurons, ReLU activation.
4. **Dense Layer 3**: 64 neurons, ReLU activation.
5. **Output Layer**: 10 neurons, softmax activation for classification.

## Training

The model is trained for 50 epochs using the Adam optimizer and Sparse Categorical Crossentropy loss function.

## Prerequisites

Before running the notebooks, make sure to install the required packages. You can use the following commands to install them:

```python
!pip install -U tensorflow_datasets
!pip install ipywidgets
```

## `MNIST_TSFlow+keras.ipynb`

This notebook is used to:

1. Load the MNIST dataset using TensorFlow Datasets (TFDS).
2. Normalize and preprocess the dataset.
3. Train a neural network model to classify MNIST digits.
4. Save the trained model to a file (`mnist_model.keras`).

### Key Code Sections

- **Dataset Preparation**: Loading and normalizing the MNIST dataset.
- **Model Definition**: Creating and compiling the Keras model.
- **Training**: Training the model on the MNIST training dataset.
- **Evaluation**: Evaluating the model on the test dataset.
- **Visualization**: Plotting some sample images and their predictions.
- **Saving**: Saving the trained model to `mnist_model.keras`.



## `ModelExecution.ipynb`

This notebook is used to:

1. Load the saved model (`mnist_model.keras`).
2. Upload images for prediction using a Colab file upload widget.
3. Process the uploaded images and make predictions.
4. Display the predicted class along with the confidence percentage.

### Key Code Sections

- **Image Upload**: Uploading images using a Colab file upload widget.
- **Image Processing**: Converting uploaded images to the format required by the model.
- **Prediction**: Making predictions using the loaded model.
- **Visualization**: Displaying the uploaded image and prediction results.


## How to Use

1. **Train and Save Model**:
   - Open and run the `MNIST_TSFlow+keras.ipynb` notebook.
   - This will train the model and save it as `mnist_model.keras`.

2. **Load Model and Predict**:
   - Open and run the `ModelExecution.ipynb` notebook.
   - Upload an image and observe the model's prediction.


### Uploading an Image

1. Run the `ModelExecution.ipynb` notebook.
2. Use the upload widget to select an image file.
3. The notebook will display the image along with the model’s prediction and confidence percentage.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
