import os
from PIL import Image
import numpy as np
from joblib import load

width=120
height=120

Autres = 'Data/Autres/'
rep = os.listdir(Autres)


def resize_image(name):
    img = Image.open(name)
    img = img.resize((width, height), Image.ANTIALIAS)
    img = img.convert('RGB')
    img.save(name)
    
def predict_main(rep):
    X = []
    clf2 = load('monclf.joblib')
    for file in rep:
        resize_image(Autres+file)
        im = Image.open(Autres+file)
        img = np.array(im)
        image = np.reshape(img, width*height*3)
        X.append(image)
    Xt = np.array(X)
    Yt = clf2.predict(Xt)
    return Yt
  
print (predict_main(rep))