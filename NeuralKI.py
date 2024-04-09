import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset (e.g., MNIST handwritten digits dataset)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the neural network architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input layer (flatten the 28x28 image)
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation function
    layers.Dropout(0.2),  # Dropout layer to prevent overfitting
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (for 10 classes) and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')





# import tensorflow as tf
# from tensorflow.keras import layers, models

# # Define the neural network architecture
# model = models.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(784,)),  # Input layer with 64 neurons and ReLU activation function
#     layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation function
#     layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (for 10 classes) and softmax activation
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Print the model summary
# model.summary()

