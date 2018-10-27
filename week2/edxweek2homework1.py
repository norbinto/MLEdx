import numpy as np
import matplotlib.pyplot as plt
# Useful module for dealing with the Gaussian density
from scipy.stats import norm, multivariate_normal

def density_plot(feature, label):
    plt.hist(trainx[trainy==label,feature], normed=True)
    #
    mu = np.mean(trainx[trainy==label,feature]) # mean
    var = np.var(trainx[trainy==label,feature]) # variance
    std = np.sqrt(var) # standard deviation
    #
    x_axis = np.linspace(mu - 3*std, mu + 3*std, 1000)
    plt.plot(x_axis, norm.pdf(x_axis,mu,std), 'r', lw=2)
    plt.title("Winery "+str(label) )
    plt.xlabel(featurenames[feature], fontsize=14, color='red')
    plt.ylabel('Density', fontsize=14, color='red')
    plt.show()

# Assumes y takes on values 1,2,3
def fit_generative_model(x,y,feature):
    k = 3 # number of classes
    mu = np.zeros(k+1) # list of means
    var = np.zeros(k+1) # list of variances
    pi = np.zeros(k+1) # list of class weights
    for label in range(1,k+1):
        indices = (y==label)
        mu[label] = np.mean(x[indices,feature])
        var[label] = np.var(x[indices,feature])
        pi[label] = float(sum(indices))/float(len(y))
    return mu, var, pi

def show_densities(feature):
    mu, var, pi = fit_generative_model(trainx, trainy, feature)
    colors = ['r', 'k', 'g']
    for label in range(1,4):
        m = mu[label]
        s = np.sqrt(var[label])
        x_axis = np.linspace(m - 3*s, m+3*s, 1000)
        plt.plot(x_axis, norm.pdf(x_axis,m,s), colors[label-1], label="class " + str(label))
    plt.xlabel(featurenames[feature], fontsize=14, color='red')
    plt.ylabel('Density', fontsize=14, color='red')
    plt.legend()
    plt.show()

def test_model(feature):
    mu, var, pi = fit_generative_model(trainx, trainy, feature)

    k = 3 # Labels 1,2,...,k
    n_test = len(testy) # Number of test points
    score = np.zeros((n_test,k+1))
    for i in range(0,n_test):
        for label in range(1,k+1):
            score[i,label] = np.log(pi[label]) + \
            norm.logpdf(testx[i,feature], mu[label], np.sqrt(var[label]))
    predictions = np.argmax(score[:,1:4], axis=1) + 1
    # Finally, tally up score
    errors = np.sum(predictions != testy)
    print ("Test error using feature " + featurenames[feature] + ": " + str(errors) + "/" + str(n_test))
    return errors

#BEGIN SOLUTION
def train_model_error_rate(feature):
    mu, var, pi = fit_generative_model(trainx, trainy, feature)

    k = 3 # Labels 1,2,...,k
    n_train = len(trainy) # Number of train points
    score = np.zeros((n_train,k+1))
    for i in range(0,n_train):
        for label in range(1,k+1):
            score[i,label] = np.log(pi[label]) + \
            norm.logpdf(trainx[i,feature], mu[label], np.sqrt(var[label]))
    predictions = np.argmax(score[:,1:4], axis=1) + 1
    # Finally, tally up score
    errors = np.sum(predictions != trainy)
    #print ("Train error using feature " + featurenames[feature] + ": " + str(errors) + "/" + str(n_train))
    return errors
#END SOLUTION

data = np.loadtxt('week2/wine.data.txt', delimiter=',')
# Names of features
featurenames = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols', 
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 
                'OD280/OD315 of diluted wines', 'Proline']

# Split 178 instances into training set (trainx, trainy) of size 130 and test set (testx, testy) of size 48
# Also split apart data and labels
np.random.seed(0)
perm = np.random.permutation(178)
trainx = data[perm[0:130],1:14]
trainy = data[perm[0:130],0]
testx = data[perm[130:178], 1:14]
testy = data[perm[130:178],0]

sum(trainy==1), sum(trainy==2), sum(trainy==3)

#BEGIN SOLUTION Problem 1
sum(testy==1), sum(testy==2), sum(testy==3)
#END SOLUTION

#BEGIN SOLUTION Problem 2
std = np.zeros(13)
for feature in range(0,13):
    std[feature] = np.std(trainx[trainy==1,feature])
print(np.argmin(std))
#END SOLUTION

feature = 0 # 'alcohol'
mu, var, pi = fit_generative_model(trainx, trainy, feature)
print(pi[1:])

#BEGIN SOLUTION PROBLEM 6 PROBLEM 7
i=0
train_error_rates=np.array([])
test_error_rates=np.array([])
while i<13:
    train_error_rates=np.insert(train_error_rates,[train_error_rates.size],train_model_error_rate(i))
    test_error_rates=np.insert(test_error_rates,[test_error_rates.size],test_model(i))
    i+=1

#number of smallest item indexes
k = 3

#collect the smallest indexes
train_error_rates_idx = np.argpartition(train_error_rates, (1,k))
test_error_rates_idx = np.argpartition(test_error_rates, (1,k))

#print the first K smallest indexes
print(str(k)+" smallest indexes for train set error rates: "+str(train_error_rates_idx[:k]))
print(test_error_rates)
print(test_error_rates_idx)
print(str(k)+" smallest indexes for test set errors rates: "+str(test_error_rates_idx[:k]))
#END SOLUTION

#stupid commit