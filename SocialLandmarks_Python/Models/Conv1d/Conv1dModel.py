from imports import *
from helper_functions import *

# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\Conv1d\Conv1dModel.py

def instantiate_model(seq_length):

    num_features = 2  
    num_classes = 36 

    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(seq_length, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ ==  '__main__':
    model = instantiate_model()

    # Gnerate dummy data:
    # seq_length = 100  
    # num_features = 2  
    # num_classes = 36 
    # num_samples = 1000
    # X = np.random.rand(num_samples, seq_length, num_features)
    # y = np.random.randint(0, num_classes, size=(num_samples,))
    # print(model.predict(X).shape)