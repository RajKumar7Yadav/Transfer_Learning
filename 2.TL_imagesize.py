# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:42:30 2022
@author: raj.yadav
"""
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

inp_size=(256,256,3)

#1. load your Vgg model
model=VGG16(include_top=False,weights='imagenet',input_shape=inp_size)#(include_top=False,weights='imagenet')
#model.summary()
"""
we see that the no of trainable parameter Vgg16 modelis 138,357,544 and input size is 224,224,3
Now we change the input size to 256,256,3. Does it work for a rectangular dim such as 280,224,3 etc? 
"""
#2. loading Vgg model with different image size

#model_resize=VGG16(input_shape=inp_size)
#model_resize.summary() throws an error as When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).  Received: input_shape=(256, 256, 3)
#orig_model_conv1_block1_wts = model.layers[1].name  #.get_weights()[0]

"""
it seems we cannot chop of the input layer from model as a default input layer gets added to it every time
as show above we cannot change input shape of a model without include_top to be false, then we  will go to
the 3rd step where we make input_top as False and then change the shape of input layer
"""
# layer_name='input_1'
# model2=Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
# model3=Model(inputs=model.get_layer(layer_name).output,outputs=model.output)
# model3.summary()


"""
what we see is that after resize, the total trainable parameter which was 138,357,544 for input size is 224,224,3
is changed to 165,758,794. This is bcz of of Dense layer at the end the trainable parameter increases, but we want the trainable parameters to be 
equal,hence we use 'include_top' as false which means import layers till flattening,and exclude dense layer
we put our own dense layer for different size
Conclusion: We cannot use different input dimension if we plan on using dense layer
"""
# a handwrittten Vgg16 model layer
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
# model = Sequential()

# model.add(Conv2D(input_shape=(256,256,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# model.add(Flatten())
# model.add(Dense(units=4096,activation="relu"))
# model.add(Dense(units=4096,activation="relu"))
# model.add(Dense(units=10, activation="softmax")) 

# print(model.summary())

# 4. include_top as False
model_resize=VGG16(input_shape=inp_size,include_top=False,weights='imagenet')
model_resize.summary()

"""
14,714,688 is the total trainable parameters without flattening/dense layer for all input dimension 
i.e(224,224,3) and (256,256,3),we can add dense layer as per our need. The image size after every 
convolution layer changes but the no of parameter doesnt change bcz the parameter hs nothing to do with
image size , it denpeds of kernel size which remains unchanged 

"""