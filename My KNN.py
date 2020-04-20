from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets , metrics
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

class MyKNearestNeighbor:

    def __init__(self, n_neighbors = 3):
        self.n_neighbors = n_neighbors
        
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train 
        
    def predict(self, x_test):
        return self.__kNearestNeighbors(self.x_train, x_test, self.y_train, self.n_neighbors)

    def __kNearestNeighbors(self, x_train, x_test, y_train, k):
        test_size = x_test.shape[0] #number of testing data 
        train_size = x_train.shape[0] #number of training data

        d = np.zeros((test_size,train_size)) #test_size = row,train_size = col

        for i in range(test_size):
            for j in range(train_size):
                d[i,j] = dist(x_train[j],x_test[i]) #d[i] array holds the destince of i'th test image and all the train image

        ind = np.zeros((test_size,k))

        for i in range(test_size):
            ind[i] = np.argpartition(d[i], k)[:k] #quickly find the k smallest elements index.not necesseryly sorted 

        y = np.zeros((test_size,k))

        ind = ind.astype(int)

        for i in range(test_size):
            y[i] = y_train[ind[i]] #finds the lables of the corrosponding indexes

        y_pred = np.zeros(test_size)

        for i in range(test_size):
           y_pred[i] = np.argmax(np.bincount(y[i].astype(np.int32))) #find the most frequent lable in this list

        y_pred = y_pred.astype(int)# froat to int

        return y_pred
    

class Main:
    
    def __init__(self, n_neighbors = 3):
        self.n_neighbors = n_neighbors
    
    def __LoadDataSet(self):
        self.digits = datasets.load_digits()
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.digits.data, self.digits.target, test_size = 0.25, random_state = 0)
        
    def __Plot(self):
        plt.matshow(self.digits.images[5])
        
        plt.matshow(self.digits.images[6]) 
        
        plt.show()
        
    def __BuildInClassifier(self):
        clf = KNeighborsClassifier(n_neighbors = self.n_neighbors)

        clf.fit(self.x_train, self.y_train)

        y_pred = clf.predict(self.x_test)

        print('sklearn built in KNN classification report:')

        print(metrics.classification_report(self.y_test, y_pred))
        
    def __MyClassifier(self):
        myClf = MyKNearestNeighbor(n_neighbors = self.n_neighbors)

        myClf.fit(self.x_train, self.y_train)

        my_y_pred = myClf.predict(self.x_test)

        print('My KNN classification report:')

        print(metrics.classification_report(self.y_test, my_y_pred))
    
    def run(self):
        self.__LoadDataSet()
        self.__Plot()
        self.__BuildInClassifier()
        self.__MyClassifier()
        
x = Main()
x.run()
