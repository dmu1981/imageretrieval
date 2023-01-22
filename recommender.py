from sklearn import svm

import os 
import glob
from PIL import Image
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm
import json
import torch
import clip
import matplotlib.pyplot as plt
import random

LATENT_DIM = 768
annoy = AnnoyIndex(LATENT_DIM, 'dot')
urls = []

class Loader:
  def __init__(self):
    pass

  def load_file(self, url):
    return Image.open(url)

  def load(self, url):
    s = url.split("://")
    protocol = s[0]
    url = s[1]

    if protocol == "file":
      return self.load_file(url)

    raise "Protocol " + protocol + " not implemented in loader"

loader = Loader()

with open("urls.json", "rt") as f:
  urls = json.loads(f.read())
annoy.load("signatures.annoy")

samples = []
classes = []

def count_nr_samples(of_class):
  cnt = 0

  for c in classes:
    if c == of_class:
      cnt += 1

  return cnt

def enough_classes():
  return count_nr_samples(0) > 2 and count_nr_samples(1) > 2

def pick_new_sample(of_class):
  # First, pick a sample we already know is of this class
  while True:
    idx = int(random.uniform(0, len(classes)))
    if classes[idx] == of_class:
      break
  
  # Find n nearest neighbors
  candidates = annoy.get_nns_by_vector(samples[idx], 512)

  # Shuffle the list
  random.shuffle(candidates)
  for candidate in candidates:
    pred = clf.predict([annoy.get_item_vector(candidate)])
    if pred == of_class:
      return candidate

  return candidates[0]


clf = None
total_predictions = 0
total_correct = 0

def query_random_image():
  global total_predictions
  global total_correct

  if clf is None or not enough_classes():
    idx = int(random.uniform(0, len(urls)))
  else:
    idx = pick_new_sample(len(classes) % 2)

  if clf is not None:
    v = [annoy.get_item_vector(idx)]
    prediction = clf.predict(v)
    if prediction == 1:
      if total_predictions > 5:
        print("My accuracy so far is {:.2f}%, i predict you will like this image... ".format(total_correct / total_predictions * 100), end="")
      total_predictions += 1

    if prediction == 0:
      if total_predictions > 5:
        print("My accuracy so far is {:.2f}%, i predict you won´t like this image... ".format(total_correct / total_predictions * 100), end="")
      total_predictions += 1
        
        
  
  img = loader.load(urls[idx])
  plt.imshow(img)
  plt.show(block=False)

  while True:
    x = input("Do you like this image? (y/n)")
    if x == 'y' or x == 'n':
      break

  if x == 'y':
    if clf is not None:
      if prediction == 1:
        total_correct += 1
    samples.append(annoy.get_item_vector(idx))
    classes.append(1)
  if x == 'n':
    if clf is not None:
      if prediction == 0:
        total_correct += 1
    samples.append(annoy.get_item_vector(idx))
    classes.append(0)


for j in range(200):
  query_random_image()

  if enough_classes():
    clf = svm.SVC()
    clf.fit(samples, classes)  

print(samples)
print(classes)