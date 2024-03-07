#part1
import numpy as np

def func(xc):
    return xc * np.sin((xc**2)/300)
#    return xc**3 - 50*xc**2 - 500*xc + 1000
#    return xc ** 2

#this is the range and the samples I want
xc = np.linspace(-100, 100, 10000)
yc = func(xc)

training_data = np.column_stack((xc, yc))
print(training_data[:10000])

#part2
#model1
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split

# Split data into train and test sets
x_train1, x_test1, y_train1, y_test1 = train_test_split(xc, yc, test_size=0.6, random_state=42)

#define model
model1 = Sequential()
model1.add(Input(shape=(1,)))
model1.add(Dense(150,activation = "relu"))
model1.add(Dense(150,activation = "relu"))
model1.add(Dense(150,activation = "relu"))
model1.add(Dense(150,activation = "relu"))
model1.add(Dense(150,activation = "relu"))
model1.add(Dense(150,activation = "relu"))
model1.add(Dense(150,activation = "relu"))
model1.add(Dense(150,activation = "relu"))
model1.add(Dense(150,activation = "relu"))
model1.add(Dense(1))

#compile model
model1.compile(
   loss = 'MAE', optimizer = 'Adam', metrics=['mae']
)
#model.fit(X_train, Y_train, batch_size = 128, epochs = 50)
model1.fit(x_train1, y_train1, epochs=50, batch_size=128, validation_data=(x_test1, y_test1))



#model2 - trying LeakyReLU to avoid dead neurons
# Split data into train and test sets
x_train2, x_test2, y_train2, y_test2 = train_test_split(xc, yc, test_size=0.6, random_state=42)

#define model
model2 = Sequential()
model2.add(Input(shape=(1,)))
model2.add(Dense(150,activation = "LeakyReLU"))
model2.add(Dense(150,activation = "LeakyReLU"))
model2.add(Dense(150,activation = "LeakyReLU"))
model2.add(Dense(150,activation = "LeakyReLU"))
model2.add(Dense(150,activation = "LeakyReLU"))
model2.add(Dense(150,activation = "LeakyReLU"))
model2.add(Dense(150,activation = "LeakyReLU"))
model2.add(Dense(150,activation = "LeakyReLU"))
model2.add(Dense(150,activation = "LeakyReLU"))
model2.add(Dense(1))

#compile model
model2.compile(
   loss = 'MAE', optimizer = 'Adam', metrics=['mae']
)
#model.fit(X_train, Y_train, batch_size = 128, epochs = 50)
model2.fit(x_train2, y_train2, epochs=50, batch_size=128, validation_data=(x_test2, y_test2))


#model3 - trying ELU

# Split data into train and test sets
x_train3, x_test3, y_train3, y_test3 = train_test_split(xc, yc, test_size=0.6, random_state=42)

#define model
model3 = Sequential()
model3.add(Input(shape=(1,)))
model3.add(Dense(150,activation = "ELU"))
model3.add(Dense(150,activation = "ELU"))
model3.add(Dense(150,activation = "ELU"))
model3.add(Dense(150,activation = "ELU"))
model3.add(Dense(150,activation = "ELU"))
model3.add(Dense(150,activation = "ELU"))
model3.add(Dense(150,activation = "ELU"))
model3.add(Dense(150,activation = "ELU"))
model3.add(Dense(150,activation = "ELU"))
model3.add(Dense(1))

#compile model
model3.compile(
   loss = 'MAE', optimizer = 'Adam', metrics=['mae']
)
#model.fit(X_train, Y_train, batch_size = 128, epochs = 50)
model3.fit(x_train3, y_train3, epochs=50, batch_size=128, validation_data=(x_test3, y_test3))




#part3
import matplotlib.pyplot as plt

# Evaluate the model1 on training data
train_loss1 = model1.evaluate(x_train1, y_train1)
print("Training Loss1:", train_loss1)

# Evaluate the model on test data
#loss1, accuracy1 = model1.evaluate(x_test1, y_test1)
test_loss1 = model1.evaluate(x_test1, y_test1)
print("Test Loss1:", test_loss1)

py1 = model1.predict(xc)

plt.plot(xc, py1)
plt.plot(xc, yc)
plt.title("Model1 Fit")
plt.show()

# Evaluate the model2 on training data
train_loss2 = model2.evaluate(x_train2, y_train2)
print("Training Loss2:", train_loss2)

# Evaluate the model on test data
test_loss2 = model2.evaluate(x_test2, y_test2)
print("Test Loss2:", test_loss2)

py2 = model2.predict(xc)

plt.plot(xc, py2)
plt.plot(xc, yc)
plt.title("Model2 Fit")
plt.show()

# Evaluate the model3 on training data
train_loss3 = model3.evaluate(x_train3, y_train3)
print("Training Loss3:", train_loss3)

# Evaluate the model on test data
test_loss3 = model3.evaluate(x_test3, y_test3)
print("Test Loss3:", test_loss3)

py3 = model3.predict(xc)

plt.plot(xc, py3)
plt.plot(xc, yc)
plt.title("Model3 Fit")
plt.show()



#part4
# to pick the model that has the highest accuracy, we need to choos the model with lowest loss.

# Get the model with the highest accuracy
best_model = model1  # Assume model1 has the highest accuracy initially
lowest_loss = test_loss1
print("model1 selected as best model")

if test_loss2 < lowest_loss:
    lowest_loss = test_loss2
    best_model = model2
    print("model2 selected as best model")

if test_loss3 < lowest_loss:
    lowest_loss = test_loss3
    best_model = model3
    print("model3 selected as best model")

weights = []
biases = []

for layer in best_model.layers:
    weights.append(layer.get_weights()[0])
    biases.append(layer.get_weights()[1])

# Choose 5 random data points from the training dataset
x_samples = x_train1[:5]
# Reshape x_samples to have dimensions (num_samples, num_features)
x_samples = np.reshape(x_samples, (x_samples.shape[0], 1))

# Compute the output of the model manually
for i, (w, b) in enumerate(zip(weights, biases)):
    x_samples = np.dot(x_samples, w) + b
    if i < len(weights) - 1:  # Apply activation function except for the last layer
        if (best_model == model1):
            x_samples = np.maximum(x_samples, 0)  # ReLU activation function
        if (best_model == model2):
            x_samples = np.maximum(0.3 * x_samples, x_samples) # LeakyReLU using default values
            
        if (best_model == model3):
            x_samples = np.where(x_samples >= 0, x_samples, 1 * (np.exp(x_samples) - 1))  # ELU activation function

# Get the output of the model using model.predict()
predicted_output = best_model.predict(x_train1[:5])

# Print the manually computed output and the output obtained from model.predict()
print("Manually Computed Output (weights and bias):")
print(x_samples)

print("\nOutput from model.predict():")
print(predicted_output)