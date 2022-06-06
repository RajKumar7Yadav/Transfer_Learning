# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:49:04 2022
@author: raj.yadav
"""

from keras.models import Model
from keras.layers import Dense
from keras.models import load_model

model=load_model("FaceQnet.h5")
#model.summary()

# check no of layers in old model
count=0
for i,layer in enumerate(model.layers):
    count=count+1
print(i)

# this method just count and create model till 2nd last, basically one layer is from below is removed
layer_name='layer_extra1'
model1=Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
#model1.summary()
#model1.save("try.h5")

prediction=Dense(5, activation='softmax')(model1.layers[-1].output)
model_new=Model(inputs=model1.input,outputs=prediction)
model_new.compile(loss='categorical_crossentropy',optimizer='adam')
model_new.summary()
#model1.save("FaceQnet_new.h5")

# Now try this with the Vgg16 model
from keras.applications.vgg16 import VGG16

model=VGG16()
model.summary()

# the prediction layer has been removed
layer_name='fc2'
model1=Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
model1.summary()

#Now we will add 2 of our own prediction layer
prediction=Dense(10,activation='softmax')(model1.layers[-1].output)
Vgg16_new=Model(inputs=model1.input,outputs=prediction)
prediction=Dense(5,activation='softmax')(Vgg16_new.layers[-1].output)
Vgg16_new=Model(inputs=model1.input,outputs=prediction)

#model_new.compile(optimizer='adam',loss='categorical_crossentropy')
Vgg16_new.summary()