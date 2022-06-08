# Transfer_Learning
The reuse of previously learned model on a new problem is called Transfer learning. The knowledge of an already trained ML model is transferred to a different but a closely linked problem. Example: a model classifying as a dog can be used for classification of a cat.
Benefits of Transfer learning is that we get a pre trained neural netowrks which has been trained with a large dataset and now can be used on samll dataset.
There are generally 3 types of transfer learning technique:-

   1. Transfer learning where we fine tune the learned model to change the output of the model. Example: A multiclass classifier can be used as Binary Classifier by changing the output layer of the model. Transfer_Learning.py is one such example.
   
   2. Transfer learning for different image size. We can change the input size of a model. Example: if a model has been trained on a images of size (244,244). We can use the model to train with a dataset of dimension (256,256) . Since the no of parameter is dependent on the kernel size and not the image size it remains unchanged. File:-2.TL_imagesize.py is one such example.
    
   3. Transfer learning for different image channel. It is also possible to take up a ML model trained on a 3 channel image(RGB) can be used to train on images which are 1-channel(Gray image).
