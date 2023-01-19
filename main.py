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

# Load the CLIP Model itself
print("Loading CLIP... please wait")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hint: The ViT-L/14@336px is the "best" model according to the original publication.
# Its latent space has 768 dimensions. There are other models with different characteristics and dimensionality.
# run clip.available_models() to get a list of all available models.
model, preprocess = clip.load("ViT-L/14@336px", device=device)
LATENT_DIM = 768

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

class Processing:
  def __init__(self):
    pass

  def process(self, image):
    return np.random.normal(0.0, 1.0, (768))

annoy = AnnoyIndex(LATENT_DIM, 'dot')
urls = []

LOADER = Loader()    
PROCESSING = Processing()

class Indexer:
  def __init__(self):
    pass

class DirectoryIndexer(Indexer):
  def __init__(self, root, extensions):
    self.root = root
    
    self.extensions = [x.lower() for x in extensions]

  def name(self):
    return "Directory: " + self.root

  def crawl(self):
    bar = tqdm(glob.iglob(self.root+"\\**", recursive=True))

    start = len(urls)
    for index, path in enumerate(bar):  
      bar.set_description("#{:6}/{:6} -> ...{}".format(len(urls), start + index, path[-80:]))

      if not os.path.isfile(path): 
        continue

      if "$RECYCLE.BIN" in path:
        continue

      filename, file_extension = os.path.splitext(path)
      if file_extension.lower() not in self.extensions:
        continue

      try:
        path = "file://" + path
        img = LOADER.load(path)
        img = preprocess(img).unsqueeze(0).to(device)
        #sig = PROCESSING.process(img)
        with torch.no_grad():
          image_features = model.encode_image(img) 
        image_features = image_features / image_features.norm(dim=1,keepdim=True)

        annoy.add_item(len(urls), image_features.view(LATENT_DIM))
        urls.append(path)
      except:
        pass

class MultipleIndexer(Indexer):
  def __init__(self, indexer):
    self.indexer = indexer
    self.iter = self.indexer.__iter__()
    self.current = None

  def crawl(self):
    for idx in self.indexer:
      print(idx.name())
      idx.crawl()
    
c = MultipleIndexer([
      DirectoryIndexer("D:\\python\\", [".png", ".jpg", ".jpeg"]),
    ])

def crawl():
  print("Crawling sources...")
  c.crawl()
  print("Building Tree")
  annoy.build(64)
  print("Saving Tree")
  annoy.save('signatures.annoy')
  with open("urls.json", "wt") as f:
    f.write(json.dumps(urls))

# Use matplotlib to display the results    
def show_best_matches(matches, query):
  fig, axs = plt.subplots(2,4)
  fig.suptitle(query)
  for index, match in enumerate(matches):
    print(match)
    pil = LOADER.load(match)
    axs[index//4][index%4].imshow(pil)
    axs[index//4][index%4].get_xaxis().set_visible(False)
    axs[index//4][index%4].get_yaxis().set_visible(False)

  plt.show()

def query():
  with open("urls.json", "rt") as f:
    urls = json.loads(f.read())
  annoy.load("signatures.annoy")


  while True:
    # Get text input from stdin
    data = input("> \n")
    if 'exit' == data:
        break
    
    # Tokeninze the input
    text = clip.tokenize([data]).to(device)
    
    # We don´t need gradients for this step
    with torch.no_grad():
      # Encode the text using the CLIP model
      text_features = model.encode_text(text) 

    # Normalize the vector for later look-up
    text_features = text_features / text_features.norm(dim=1,keepdim=True)
    
    # Match it against our database
    #text_features = text_features.cpu().numpy()

    indices = annoy.get_nns_by_vector(text_features.view(LATENT_DIM), 8)
    paths = [urls[idx] for idx in indices]

    show_best_matches(paths, data)

crawl()
#query()
