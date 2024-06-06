from imports import *


def instantiate_model():
    model = Sequential()
    
    # First Convolutional Block
    model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 1), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    
    # Second Convolutional Block
    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(2048, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1024, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(25))
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