import tensorflow
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D,Input,MaxPooling2D,Dense,Activation,Flatten,Dropout
from keras.utils import np_utils



ds = pd.read_csv('fashionMNIST.csv')
ds = ds.values
ds =np.array(ds)
print(ds.shape)

X_ = ds[:,1:]
Y_ = ds[:,0]
print(X_.shape)
print(Y_.shape)

x_train = X_.reshape(10000,28,28,1)
y_train = np_utils.to_categorical(Y_)
x_train=np.array(x_train)
y_train=np.array(y_train)
print(x_train.shape)
print(y_train.shape)

for ix in range(1,10):
    plt.figure(ix)
    plt.imshow(x_train[ix].reshape(28,28), cmap='gray')
    plt.show()
    # x_train[ix] = x_train[ix].reshape(28,28,1)
	


# print(x_train[1].reshape(28,28))


# CNN Model
model = Sequential()
model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(32,(5,5),activation='relu'))
model.add(Convolution2D(8,(5,5),activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(x_train,y_train
                    ,epochs=10,validation_split=0.2
                    ,shuffle=True,batch_size=256)


plt.figure(0)
plt.plot(history.history['loss'],'g',label='loss')
plt.plot(history.history['val_loss'],'b',label='Val_loss')
plt.legend()
plt.show()

plt.figure(0)
plt.plot(history.history['acc'],'g',label='acc')
plt.plot(history.history['val_acc'],'b',label='Val_acc')
plt.legend()
plt.show()

# Saving the model to save training everytime
# model.save('model_cnn.h5')
