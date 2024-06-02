from imports import *

def instantiate_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 1)))
    model.add(BatchNormalization())  # BatchNormalization after Conv2D
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())  # BatchNormalization after Conv2D
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(BatchNormalization())  # BatchNormalization after Dense
    model.add(Activation('relu'))
    model.add(Dropout(0.2)) 

    model.add(Dense(1024))
    model.add(BatchNormalization())  # BatchNormalization after Dense
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(BatchNormalization())  # BatchNormalization after Dense
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(36))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ ==  '__main__':
    model = instantiate_model()
    
    # #inputs= torch.randn(batch_size, 32, 32,requires_grad=True)
    # inputs = np.random.random((batch_size, 32, 32, 1))
    # outputs =  model(inputs)
    # print(outputs.shape)
    # exit()