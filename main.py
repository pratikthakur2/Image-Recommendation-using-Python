import numpy as np
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential
from keras.layers  import Dense, GlobalMaxPooling2D
import cv2
# from sklearn.cluster import KMeans
import pickle

model = VGG16(weights='imagenet',include_top=False,input_shape=(244,244,3))
model.trainable = False

model = Sequential([model,GlobalMaxPooling2D()])

def extract_features(image_path,model):
  img=cv2.imread(image_path)
  img_array=np.expand_dims(cv2.resize(img,(244,244)),axis=0)
  process_image=preprocess_input(img_array)
  result=model.predict(process_image).flatten()
  normalize=result/np.linalg.norm(result)
  return normalize

def cosine_similarity(v1,v2):
    v1 = np.array(v1).flatten()
    v2 = np.array(v2).flatten()
    return v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2))

images =pickle.load(open('Image_Embedding.pkl','rb'))
labels =pickle.load(open('image_cluster_labels.pkl','rb'))
kmeans_model =pickle.load(open('kmeans.pkl','rb'))



label_to_image = {}
for i in range(len(labels)):
   label = labels[i]
   img = images[i]
   if label in label_to_image:
      label_to_image[label].append(img)
   else:
      label_to_image[label]= [img]


def find_similar(image):
   similarity = {}
   image = image.astype(float)
   target_label = kmeans_model.predict(image.reshape(1,-1))[0]
   target_img = label_to_image[target_label]

   for img in range(len(target_img)):
      similarity[img] = cosine_similarity(target_img[img],image)
   return sorted(similarity.items(), key = lambda x:x[1], reverse=True)[:5], target_label
