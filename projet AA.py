import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#accuracy_score(y_test,y_predits)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from scipy import misc
#from sklearn.neighbors import NeighborhoodComponentsAnalysis
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron



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

for name in files1:
    resize_image(Mer+name)
    rotate_image(Mer+name)
    im = Image.open(Mer+name)
    img = np.array(im)
    niquel = np.reshape(img, longueur*largeur*3)   
    X.append(niquel)
    y.append(1)
    
for name in files2:
    resize_image(Ailleurs+name)
    rotate_image(Ailleurs+name)
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
#
#print("_____now let's test with k_neighbor_____")
#for n in range (4,10):
#    print("nb voisins: ", n)
#    knn = KNeighborsClassifier(n_neighbors=n)
#    knn.fit(X_train, y_train)
#    print(knn.score(X_train, y_train))
#    print(knn.score(X_test, y_test))
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
#logicR = LogisticRegression(random_state=0).fit(X_train, y_train)
#logicR.predict(X_train)
#logicR.predict_proba(X_train)
#print(logicR.score(X_train, y_train))
#print(logicR.score(X_test, y_test))