
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Activation, Dropout

def Model(input_shape):
    
    model = Sequential()
    
    #First two conV layers used 64 filter
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu') )
    
    model.add( Conv2D(64, (3, 3), activation='relu', padding='same') )
    model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) )
    
    #Second conV layers used 128 filter
    model.add( Conv2D(128, (3, 3), activation='relu', padding='same') )
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same') )
    model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) )
    
    #Third conV layers used 256 filter
    model.add(  Conv2D(256, (3, 3), activation='relu', padding='same') )
    model.add(  Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(  Conv2D(256, (3, 3), activation='relu', padding='same') )
    model.add(  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) )
    
    #Fourth conV layers used 512 filter
    model.add(  Conv2D(512, (3, 3), activation='relu', padding='same') )
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same') )
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) )
    
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same') )
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add( MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(4096, activation='relu'))
    model.add( Dense(4096, activation='relu'))
    model.add( Dense(1000, activation='softmax'))
    
    return model

model = Model( (224, 224, 3))
model.summary()


