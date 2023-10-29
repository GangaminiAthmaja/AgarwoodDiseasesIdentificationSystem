#!/usr/bin/env python
# coding: utf-8

# ### Import all Libraies

# In[1]:


import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML


# ### Set all the Constants

# In[2]:


IMAGE_SIZE = 256
BATCH_SIZE = 32 #standerd batch size is 32
CHANNELS = 3
EPOCHS = 50


# ### Import data into tensorflow dataset object

# We will use image_dataset_from_directory api to load all images in tensorflow dataset
# 
# This function is used to create a dataset of images from a directory.
# function takes several parameters to customize the behavior of the dataset creation.

# In[3]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "AgarwoodPlantDiseases",
    shuffle = True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE
)


# In[4]:


class_names = dataset.class_names
print(class_names)


# In[5]:


len(dataset)


# In[6]:


for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())


# As you can see above, each element in the dataset is a tuple. First element is a batch of 32 elements of images. Second element is a batch of 32 elements of class labels

# In[7]:


for image_batch, label_batch in dataset.take(1):
    print(image_batch[0].shape)


# In[8]:


for image_batch, label_batch in dataset.take(1):
    print(image_batch[0].numpy())


#### Visualize some of the images from our dataset

# In[9]:


plt.figure(figsize=(10, 10))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off") 


# Dataset should be bifurcated into 3 subsets, namely:
# 
# 1. Training: Dataset to be used while training
# 2. Validation: Dataset to be tested against while training
# 3. Test: Dataset to be tested against after we trained a model

# In[10]:


len(dataset)


# ### Function to Split Dataset

# 80% ==> training 20% ==> 10% validation, 10% test

# In[11]:


train_size = 0.8
len(dataset)*train_size


# In[12]:


#that train_ds will contain 54 batches of images and labels from the original dataset.
train_ds = dataset.take(54)
len(train_ds)


# In[13]:


#that test_ds will contain the remaining batches of images and labels from the original dataset, starting from the 55th batch onwards.
test_ds = dataset.skip(54)
len(test_ds)


# In[14]:


val_size = 0.1
len(dataset)*val_size


# In[15]:


val_ds = test_ds.take(6)
len(val_ds)


# In[16]:


test_ds = test_ds.skip(6)
len(test_ds)


# In[17]:


#The function allows for customization of the split ratios and provides options for shuffling the dataset.
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size = 10000):
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)
        
    train_size =int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[18]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[19]:


len(train_ds)


# In[20]:


len(val_ds)


# In[21]:


len(test_ds)


# Cache
# improve training performance, 
# Caching allows faster access to the data during training as it avoids unnecessary re-reading or reprocessing of the data.
# 
# shuffle
# This method shuffles the elements of the dataset
# 
# prefetch
# 
# This method adds a prefetching step to the dataset. Prefetching allows the GPU or CPU to overlap the data preprocessing and model execution. The buffer_size parameter determines the number of elements to prefetch.
# 
# 
# 
# #applying additional transformations to optimize its performance during training.

# ### Cache, Shuffle, and Prefetch the Dataset

# In[22]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# ### Building the Model

# Before we feed our images to network, we should be resizing it to the desired size. Moreover, to improve model performance, we should normalize the image pixel value (keeping them in range 0 and 1 by dividing by 256). This should happen while training as well as inference. Hence we can add that as a layer in our Sequential Model.
# 
# You might be thinking why do we need to resize (256,256) image to again (256,256). You are right we don't need to but this will be useful when we are done with the training and start using the model for predictions. At that time somone can supply an image that is not (256,256) and this layer will resize it

# ### Creating a Layer for Resizing and Normalization

# In[23]:


resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])


# Data Augmentation is needed when we have less data, this boosts the accuracy of our model by augmenting the data.

# ### Data Augmentation

# In[24]:


data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])


# ### Applying Data Augmentation to Train Dataset

# In[25]:


train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# ### Model Architecture

# We use a CNN coupled with a Softmax activation in the output layer. We also add the initial layers for resizing, normalization and Data Augmentation.
# 
# We are going to use convolutional neural network (CNN) here.

# In[26]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 6

model = models.Sequential([ 
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax'),  # Change the number of units to 6
])



model.build(input_shape=input_shape)


# In[27]:


model.summary()


# ### Compiling the Model

# We use adam Optimizer, SparseCategoricalCrossentropy for losses, accuracy as a metric

# In[28]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[29]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=50,
)


# In[31]:


scores = model.evaluate(test_ds)


# In[32]:


scores


# In[33]:


history


# In[34]:


history.params


# In[35]:


history.history.keys()


# In[36]:


type(history.history['loss'])


# In[37]:


len(history.history['loss'])


# In[38]:


history.history['loss'][:5] # show loss for first 5 epochs


# In[39]:


history.history['accuracy']


# In[40]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[41]:


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[42]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])


# In[43]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[44]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[45]:


plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")


# In[46]:


import os
model_version=max([int(i) for i in os.listdir("../models") + [0]])+1
model.save(f"../models/{model_version}")


# In[47]:


model.save("agarwood.h5")


# In[ ]:




