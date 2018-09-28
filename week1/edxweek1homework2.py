import numpy as np

### BEGIN SOLUTION Nearest neighbor classification with L2 distance
def squared_dist(x,y):
    return np.sum(np.square(x - y))

## Takes a vector x and returns the index of its nearest neighbor in train_data
def find_NN_L2(x,trainx,trainy):
    # Compute distances from x to every row in train_data
    distances = [squared_dist(x,trainx[i,]) for i in range(len(trainy))]
    # Get the index of the smallest distance
    return np.argmin(distances)

def NN_L2(trainx, trainy, testx):
    # inputs: trainx, trainy, testx <-- as defined above
    # output: an np.array of the predicted values for testy
    ret = np.array([])
    for x in testx:
        index = find_NN_L2(x,trainx,trainy)
        ret = np.insert(ret,[ret.size],trainy[index])
            
    return ret
### END SOLUTION

### BEGIN SOLUTION Nearest neighbor classification with L1 distance
def l1_dist(x,y):
    return sum(abs(a - b) for a,b in zip(x,y))

## Takes a vector x and returns the index of its nearest neighbor in train_data
def find_NN_L1(x,trainx,trainy):
    # Compute distances from x to every row in train_data
    distances = [l1_dist(x,trainx[i,]) for i in range(len(trainy))]
    # Get the index of the smallest distance
    return np.argmin(distances)

def NN_L1(trainx, trainy, testx):
    # inputs: trainx, trainy, testx <-- as defined above
    # output: an np.array of the predicted values for testy
    ret = np.array([])
    for x in testx:
        fos = find_NN_L1(x,trainx,trainy)
        ret = np.insert(ret,[ret.size],trainy[fos]) 
        
    return ret
### END SOLUTION

### BEGIN SOLUTION Test errors and the confusion matrix
def confusion(testy,testy_fit):
    # inputs: the correct labels, the fitted NN labels
    # output: a 3x3 np.array representing the confusion matrix as above
    ret = np.array([[0,0,0],[0,0,0],[0,0,0]])
    i = 0
    while i < len(testy):
        ret[int(testy[i]),int(testy_fit[i])]+=1
        i+=1
    return ret
### END SOLUTION


# Load data set and code labels as 0 = ’NO’, 1 = ’DH’, 2 = ’SL’
labels = [b'NO', b'DH', b'SL']
data = np.loadtxt('week1/column_3C.dat', converters={6: lambda s: labels.index(s)})

# Separate features from labels
x = data[:,0:6]
y = data[:,6]

# Divide into training and test set
training_indices = list(range(0,20)) + list(range(40,188)) + list(range(230,310))
test_indices = list(range(20,40)) + list(range(188,230))

trainx = x[training_indices,:]
trainy = y[training_indices]
testx = x[test_indices,:]
testy = y[test_indices]

testy_L2 = NN_L2(trainx, trainy, testx)
print("retlen:" + str(len(testy_L2)))
assert(type(testy_L2).__name__ == 'ndarray')
assert(len(testy_L2) == 62) 
assert(np.all(testy_L2[50:60] == [0.,  0.,  0.,  0.,  2.,  0.,  2.,  0.,  0.,  0.]))
assert(np.all(testy_L2[0:10] == [0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.]))

testy_L1 = NN_L1(trainx, trainy, testx)
testy_L2 = NN_L2(trainx, trainy, testx)

assert(type(testy_L1).__name__ == 'ndarray')
assert(len(testy_L1) == 62) 
assert(not all(testy_L1 == testy_L2))
assert(all(testy_L1[50:60] == [0.,  2.,  1.,  0.,  2.,  0.,  0.,  0.,  0.,  0.]))
assert(all(testy_L1[0:10] == [0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.]))

L1_neo = confusion(testy, testy_L1) 
assert(type(L1_neo).__name__ == 'ndarray')
assert(L1_neo.shape == (3,3))
assert(np.all(L1_neo == [[16.,  2.,  2.],[10.,  10.,  0.],[0.,  0.,  22.]]))
L2_neo = confusion(testy, testy_L2)  
assert(np.all(L2_neo == [[17.,  1.,  2.],[10.,  10.,  0.],[0.,  0.,  22.]]))