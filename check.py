from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D

# Create and save a dummy image model
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    Flatten(),
    Dense(1, activation='sigmoid')
])
model.save("braintumor.h5")