#  Digit Classification with Deep Learning
![1_SGPGG7oeSvVlV5sOSQ2iZw](https://github.com/Rutuja-Salunke/Digit-Classification-Deep-Learning/assets/102023809/31f5d235-fd05-4ac0-b76d-8ccfe69e55bc)


- **Project Overview:**
  - Digit classification using deep learning with TensorFlow and Keras.
  - Utilizes the MNIST dataset of 28x28 grayscale images of handwritten digits (0-9).

- **Steps:**
  - **1. Import Libraries:**
    - TensorFlow and Keras for deep learning.
    - MNIST dataset for digit images.
    - `to_categorical` for one-hot encoding labels.

  - **2. Load and Preprocess Data:**
    - Load MNIST dataset and normalize pixel values.
    - Reshape images for the model.
    - One-hot encode labels.

  - **3. Build Neural Network Model:**
    - Sequential model with Conv2D, MaxPooling2D, Flatten, Dense layers.
    - Convolutional layers for feature extraction, max pooling for dimension reduction, and dense layers for classification.

  - **4. Compile Model:**
    - Use Adam optimizer, categorical crossentropy loss, and accuracy as the metric.

  - **5. Train Model:**
    - Train on the training set for 5 epochs with a batch size of 64.
    - Validate with 20% of the data.

  - **6. Evaluate Model:**
    - Evaluate the trained model on the test set.
    - Print test accuracy.

- **Usage:**
  - Train the model using the provided code.
  - Evaluate the model's performance on new digit images.
  - Adapt hyperparameters for specific use cases.

- **Dependencies:**
  - TensorFlow 2.x
  - Keras
  - NumPy

