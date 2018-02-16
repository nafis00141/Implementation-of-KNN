from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets , metrics
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np


def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def KNearestNeighbors(x_train,x_test,y_train,k):
    
    test_size = x_test.shape[0] #number of testing data 
    train_size = x_train.shape[0] #number of training data
    
    #print("test size "+ repr( test_size) +" train_size "+ repr(train_size))
    
    d = np.zeros((test_size,train_size)) #test_size = row,train_size = col
    
    for i in range(test_size):
        for j in range(train_size):
            d[i,j] = dist(x_train[j],x_test[i]) #d[i] array holds the destince of i'th test image and all the train image
            
    #print(d)
    
    ind = np.zeros((test_size,k))
    
    for i in range(test_size):
        ind[i] = np.argpartition(d[i], k)[:k] # quickly find the k smallest elements index.not necesseryly sorted 
    
    y = np.zeros((test_size,k))
    
    ind = ind.astype(int)
    
    #print(ind)
    
    #print(y_train)
    
    for i in range(test_size):
        y[i] = y_train[ind[i]] #finds the lables of the corrosponding indexes
        
    #print(y)
    
    y_pre = np.zeros(test_size)
    
    for i in range(test_size):
       y_pre[i] = np.argmax(np.bincount(y[i].astype(np.int32))) #find the most frequent lable in this list
       
    y_pre = y_pre.astype(int)# froat to int

    #print(y_pre)
    
    return y_pre
    

digits = datasets.load_digits()

'''
plt.matshow(digits.images[5]) 
plt.show() 
print(digits.target[5])
print(digits.data[5])
'''

X = digits.data
y = digits.target

'''
plt.matshow(digits.images[5]) 
print(digits.target[5])
plt.matshow(digits.images[6]) 
print(digits.target[6])
print(dist(X[5],X[6]))
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


clf = KNeighborsClassifier(n_neighbors=3)
    
clf.fit(X_train,y_train)
    
y_pred = clf.predict(X_test)
    
print('sklearn built in KNN classification report:')
    
print(metrics.classification_report(y_test, y_pred))
    
    
my_y_pred = KNearestNeighbors(X_train,X_test,y_train,3)
    
print('My KNN classification report:')
    
print(metrics.classification_report(y_test, my_y_pred))

