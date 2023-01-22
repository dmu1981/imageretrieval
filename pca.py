from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib
import os 
import glob
from PIL import Image
import PIL
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

def get_list_of_images(n_clusters, k_per_cluster, pca):
  indices = []
  coos = []
  classes = []
  for i in range(n_clusters):
    index = int(random.uniform(0, len(urls)))
    vectors = annoy.get_nns_by_item(index, 8 * k_per_cluster)
    random.shuffle(vectors)
    vectors = vectors[:k_per_cluster]
    
    for idx in vectors:    
      vec = annoy.get_item_vector(idx)
      coordinates = pca.transform([vec])[0]
      coos.append([coordinates[0], coordinates[1], coordinates[2]])
      classes.append(i)
      
    indices += vectors

  return indices, coos, classes

X = []
print("Preprocessing")
## Calculate PCA for all vectors
for i in tqdm(range(len(urls))):
  X.append(annoy.get_item_vector(i))

## Calculate PCA only for a subset
#index = int(random.uniform(0, len(urls)))
#vectors = annoy.get_nns_by_item(index, 4096)
#X = [annoy.get_item_vector(i) for i in vectors]
#indices = vectors

print("PCA")
pca = PCA(n_components=3)
pca.fit(X)

#coos = []
#for idx in vectors:    
  #vec = annoy.get_item_vector(idx)
  #coordinates = pca.transform([vec])[0]
  #coos.append([coordinates[0], coordinates[1], coordinates[2]])

indices, coos, classes = get_list_of_images(6, 128, pca)

# Load all images
print("Loading images")
images = [loader.load(urls[idx]) for idx in indices]
images = [x.resize(size=(128,128), resample=PIL.Image.BICUBIC) for x in images]

### Build our super-texture
# canvas = PIL.Image.new("RGB", size=(4096, 4096))
# for idx in range(len(indices)): 
#   row = idx // 32
#   col = idx % 32
#   x = col * 128
#   y = row * 128
#   canvas.paste(images[idx], (x,y))

# canvas.save("super.png")
# with open("coordinates.json", "wt") as f:
#   f.write(json.dumps(coos))

# plt.imshow(canvas)
# plt.show()
# exit()

### Draw them in PCA coordinates
CANVAS_WIDTH = 4096
CANVAS_HEIGHT = 2048
canvas = PIL.Image.new("RGB", size=(CANVAS_WIDTH, CANVAS_HEIGHT))
X = [c[0] for c in coos]
Y = [c[1] for c in coos]
minx = min(X) - 0.1
miny = min(Y) - 0.1
maxx = max(X) + 0.1
maxy = max(Y) + 0.1
for idx in range(len(indices)): 
  xcoo = (int)((X[idx]-minx)/(maxx-minx) * CANVAS_WIDTH)
  ycoo = (int)((Y[idx]-miny)/(maxy-miny) * CANVAS_HEIGHT)
  canvas.paste(images[idx], (xcoo, ycoo))

plt.imshow(canvas)
plt.show()

#colors = ['red','green','blue','purple', 'cyan', 'black']
#plt.scatter(X, Y, c=classes, cmap=matplotlib.colors.ListedColormap(colors))
#plt.show()
