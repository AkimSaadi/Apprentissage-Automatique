import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from joblib import dump

Mer = 'Data/Mer/'
Ailleurs = 'Data/Ailleurs/'
longueur, largeur = 120,120

files1 = os.listdir(Mer)
files2 = os.listdir(Ailleurs)
X = []
y = []

def rotate_image(name):
    img = Image.open(name)
    img_rotate_1 = img.rotate(90)
    img_rotate_2 = img.rotate(180)
    img_rotate_3 = img.rotate(270)
    new_name = os.path.splitext(name)[0]
    img_rotate_1.save(new_name+'_rotate_1.jpeg')
    img_rotate_2.save(new_name+'_rotate_2.jpeg')
    img_rotate_3.save(new_name+'_rotate_3.jpeg')
    img.save(name)
def resize_image(name):
    img = Image.open(name)
    img = img.resize((longueur, largeur), Image.ANTIALIAS)
    img = img.convert('RGB')
    img.save(name)
def transpose_image(name):
    img=Image.open(name)
    img_transpose_1 = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_transpose_2 = img.transpose(Image.FLIP_TOP_BOTTOM)
    new_name = os.path.splitext(name)[0]
    img_transpose_1.save(new_name+'_transpose_1.jpeg')
    img_transpose_2.save(new_name+'_transpose_2.jpeg')
    img.save(name)
def quantite(tri1,tri2):
    d = (tri1[0]-tri2[0])**2+(tri1[1]-tri2[1])**2+(tri1[2]-tri2[2])**2
    return d
def contours (name):
    im = Image.open(name)
    image_contours = Image.new('RGB', (longueur,largeur))
    for x in range (1,longueur-1) :
        for y in range (1,largeur-1) :
            p = im.getpixel((x,y))
            p1=im.getpixel((x-1,y-1))
            q1=quantite(p,p1)
            p2=im.getpixel((x-1,y))
            q2=quantite(p,p2)
            p3=im.getpixel((x-1,y+1))
            q3=quantite(p,p3)
            p4=im.getpixel((x,y-1))
            q4=quantite(p,p4)
            p5=im.getpixel((x,y+1))
            q5=quantite(p,p5)
            p6=im.getpixel((x+1,y-1))
            q6=quantite(p,p6)
            p7=im.getpixel((x+1,y))
            q7=quantite(p,p7)
            p8=im.getpixel((x+1,y+1))
            q8=quantite(p,p8)
            if q1+q2+q3+q4+q5+q6+q7+q8<2400 :
                image_contours.putpixel((x,y),(255,255,255))
            else :
                image_contours.putpixel((x,y),(0,0,0))
    new_name = os.path.splitext(name)[0]
    image_contours.save(new_name+'_contours.jpeg')
def gris (name):
    im = Image.open(name)
    image_contours = Image.new('RGB', (longueur,largeur))
    for x in range (longueur) :
        for y in range (largeur) :
            pixel=im.getpixel((x,y))
            gris = int((pixel[0]+pixel[1]+pixel[2])/3)
            p = (gris,gris,gris)
            image_contours.putpixel((x,y),p)
    new_name = os.path.splitext(name)[0]
    image_contours.save(new_name+'_n&b.jpeg')
    
for name in files1:
    resize_image(Mer+name)
    contours(Mer+name)
    gris(Mer+name)
    rotate_image(Mer+name)
    transpose_image(Mer+name)
    
for name in files2:
    resize_image(Ailleurs+name)
    contours(Ailleurs+name)
    gris(Ailleurs+name)
    rotate_image(Ailleurs+name)
    transpose_image(Ailleurs+name)
    
print('--------------')
    
files1 = os.listdir(Mer)
files2 = os.listdir(Ailleurs)

for name in files1 :
    im = Image.open(Mer+name)
    img = np.array(im)
    niquel = np.reshape(img, longueur*largeur*3)   
    X.append(niquel)
    y.append(1)

for name in files2 :
    im = Image.open(Ailleurs+name)
    img = np.array(im)
    niquel = np.reshape(img, longueur*largeur*3)
    X.append(niquel)
    y.append(-1)
   

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("_________let's test with naives Bayes_____")
classifieur = GaussianNB()
classifieur.fit(X_train, y_train)
print(classifieur.score(X_train, y_train))
score = classifieur.score(X_test, y_test)
print(score)
dump(classifieur,'monclf.joblib')
#print("_____now let's test with k_neighbor_____")
#n=9
#print("nb voisins: ", n)
#knn = KNeighborsClassifier(n_neighbors=n)
#knn.fit(X_train, y_train)
#print(knn.score(X_train, y_train))
#print(knn.score(X_test, y_test))
#print("_____now let's test with neuronal network_____")
#
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=1)
#
#clf.fit(X_train, y_train)
#print(clf.score(X_train, y_train))
#print(clf .score(X_test, y_test))
#
#print("_____now let's test with perceptron_____")
#
#perx = Perceptron(tol=1e-3, random_state=0)
#perx.fit(X_train, y_train)
#print(perx.score(X_train, y_train))
#print(perx.score(X_test, y_test))
#
#print("_____now let's test with decision tree_____")
#
#Rtree = tree.DecisionTreeClassifier()
#Rtree = Rtree.fit(X_train, y_train)
#print(Rtree.score(X_train, y_train))
#print(Rtree.score(X_test, y_test))
#
#print("_____now let's test with logistic regression_____")
#
#logicR = LogisticRegression(solver='lbfgs',random_state=0).fit(X_train, y_train)
#logicR.predict(X_train)
#logicR.predict_proba(X_train)
#print(logicR.score(X_train, y_train))
#print(logicR.score(X_test, y_test))