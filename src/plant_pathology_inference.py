#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tkinter as tk
from tkinter import filedialog

from IPython.display import SVG


# In[2]:


#set num of output classes
train_labels = 4

#load image function
def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#load model
model = tf.keras.Sequential([DenseNet121(input_shape=(512, 512, 3),
                                             weights='imagenet',
                                             include_top=False),
                                 L.GlobalAveragePooling2D(),
                                 L.Dense(train_labels,
                                         activation='softmax')])
        
model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['categorical_accuracy'])
model.summary()
model.load_weights('../weights/20210405-232711--30b7fd1c-69ed-4482-9736-eaa29729d7ce')
SVG(tf.keras.utils.model_to_dot(Model(model.layers[0].input, model.layers[0].layers[13].output), dpi=70).create(prog='dot', format='svg'))
SVG(tf.keras.utils.model_to_dot(model, dpi=70).create(prog='dot', format='svg'))


# In[3]:


#define prediction functions
def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)

def predict(img):
    return model.layers[2](model.layers[1](model.layers[0](process(img)))).numpy()[0]


# In[5]:


#predict on test images
root = tk.Tk()
root.withdraw()

img_path = filedialog.askopenfilename()
preds = predict(load_image(img_path))

print(preds)

fig = make_subplots(rows=1, cols=2)

colors = {"Healthy":px.colors.qualitative.Plotly[0], "Scab":px.colors.qualitative.Plotly[0], "Rust":px.colors.qualitative.Plotly[0], "Multiple diseases":px.colors.qualitative.Plotly[0]}
print (colors)
if list.index(preds.tolist(), max(preds)) == 0:
    pred = "Healthy"
if list.index(preds.tolist(), max(preds)) == 1:
    pred = "Multiple diseases"
if list.index(preds.tolist(), max(preds)) == 2:
    pred = "Rust"
if list.index(preds.tolist(), max(preds)) == 3:
    pred = "Scab"
    
colors[pred] = px.colors.qualitative.Plotly[2]
print (colors)
colors = [colors[key] for key in colors.keys()]
print (colors)
fig.add_trace(go.Image(z=cv2.resize(load_image(img_path), (205, 150))), row=1, col=1)
fig.add_trace(go.Bar(x=["Healthy", "Multiple diseases", "Rust", "Scab"], y=preds, marker=dict(color=colors)), row=1, col=2)
fig.update_layout(height=400, width=800, title_text="DenseNet Predictions", showlegend=False)
fig.update_layout(template="plotly_white")


# In[ ]:



