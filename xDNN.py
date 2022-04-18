# Jack Pistagnesi, Omar Sabbagh

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from scipy.special import softmax
import math


#Classifier
def xDNN(GlobalFeature, ImageData, Label):
    
    #Prototype Identification
    prototypes = {}
    labelMax = max(Label)
    dat = {}
    imageData = {}
    
    for i in range(0, labelMax + 1):
        imageData[i] = {}
        s = np.argwhere(Label == i)
        dat[i] = GlobalFeature[s,]
        for j in range(0, len(s)):
            imageData[i][j] = ImageData[s[j][0]]
        
    # Details for referenced learning procedure steps/equations/variables can be found at https://arxiv.org/pdf/1912.02523.pdf    
    # Begin xDNN learning procedure
    for y in range(0, labelMax + 1):
        
        # Learning Procedure 1: Read first image
        Data = dat[y]
        image = imageData[y]
        
        # Learning Procedure 2: Set variables
        prototype = {}
        prototype[1] = image[0]
        featureVectAmount = np.shape(Data)[0]
        data = Data
        centre = data[0,]
        globalMean = centre
        support = np.array([1])
        rad = math.sqrt(2 - 2 * math.cos(math.pi / 6))
        radius = np.array([rad])
        n = 1
        x = np.sum(np.power(centre, 2))
        
        # Learning Procedure 3
        for i in range(2, featureVectAmount + 1):
            
            # Learning Procedure 4: Density layer. Find global mean with equation 7. Centre density and centre density 
            # max/min made for Prototypes Layer.
            globalMean = (i - 1) / i * globalMean + data[i - 1,] / i
            centreDensity = np.sum((centre - np.kron(np.ones((n,1)), globalMean)) ** 2, axis = 1)
            centreDenseMax = max(centreDensity)
            centreDenseMin = min(centreDensity)
            DataDensity = np.sum((data[i-1,] - globalMean) ** 2)
            
            # Find Euclidean distance for data density, use to find data value/position
            if i == 2:
                distance = cdist(data[i-1,].reshape(1,-1),centre.reshape(1,-1),'euclidean')[0]
            else:
                distance = cdist(data[i-1,].reshape(1,-1),centre,'euclidean')[0]
            position = distance.argmax(0)
            val = distance.max(0) ** 2
            
            # Prototypes layer.
            # Learning Procedure 6. if Equation 12 holds then...
            if DataDensity > centreDenseMax or DataDensity < centreDenseMin or val > 2 * radius[position]:
                
                # Learning Procedure 7. Create centre, prototype, support and radius according to Equation 13
                prototype[n] = image[i - 1]
                centre = np.vstack((centre, data[i-1,]))
                support = np.vstack((support, 1))
                radius = np.vstack((radius, rad))
                n = n + 1
                x = np.vstack((x,1))
            else:
                # Learning Procedure 8,9,10. Assign data to nearest prototype in data space using equation 11 
                # and update rule according to equation 14
                centre[position,] = centre[position,] * (support[position] / support[position] + 1) + data[i - 1] / (support[position] + 1)
                support[position] = support[position] + 1
                radius[position] = 0.5 * radius[position] + 0.5 * (x[position,] - sum(centre[position,] ** 2)) / 2  
        proto = {}
        proto['centre'] =  centre
        proto['Prototype'] = prototype
        prototypes[y] = proto
    return prototypes



# In charge of forming the decision by assigning labels to the validation images based on 
# the degree of similarity of the prototypes
def DecisionMakingLayer(parameters,data):
    estLabels = np.zeros((np.shape(data)[0]))
    newClass = parameters['newClassTotal'] 
    scores = np.zeros((np.shape(data)[0],newClass))
    for i in range(1, np.shape(data)[0] + 1):
        val = np.zeros((newClass, 1))
        for k in range(0, newClass):
            distance = np.sort(cdist(data[i - 1,].reshape(1, -1),parameters['parameters'][k]['centre'],'minkowski'))[0]
            val[k] = distance[0]
        val = softmax(-1 * val ** 2).T
        scores[i - 1,] = val
        estLabels[i - 1] = np.argsort(val[0])[:: -1][0]
    lab = np.zeros((newClass, 1))
    for i in range(0, newClass): 
        lab[i] = np.unique(parameters['memLabels'][i])
    estLabels = lab[estLabels.astype(int)]   
    decision = {}
    decision['estLabels'] = estLabels
    decision['scores'] = scores
    return decision


def xDNNVal(Input):
    data = Input['features']
    parameters = Input['xDNNParms']
    results = DecisionMakingLayer(parameters,data)
    valOutput = {}
    valOutput['estLabel'] = results['estLabels']
    valOutput['scores'] = results['scores']
    return valOutput


def xDNNLearn(Input):
    prototypes = xDNN(Input['features'], Input['images'], Input['labels'])
    learnOutput = {}
    learnOutput['xDNNParms'] = {}
    learnOutput['xDNNParms']['parameters'] = prototypes
    memLabels = {}
    for i in range(0, max(Input['labels']) + 1):
        memLabels[i] = Input['labels'][Input['labels'] == i] 
    learnOutput['xDNNParms']['memLabels'] = memLabels
    learnOutput['xDNNParms']['newClassTotal'] = max(Input['labels']) + 1
    return learnOutput






# Get CSV files created from Feature Extraction Layer (VGG16).
trainX = np.genfromtxt(r'trainX.csv', delimiter=',')
trainY = pd.read_csv(r'trainY.csv', delimiter=',',header=None)
testX = np.genfromtxt(r'testX.csv', delimiter=',')
testY = pd.read_csv(r'testY.csv', delimiter=',',header=None)


print("Train X Shape: ",trainX.shape)
print("Train Y Shape: ",trainY.shape)
print("Test X Shape: ",testX.shape)
print("Test Y Shape: ",testY.shape)

print("Training Model...")

# Get labels/images from CSV and convert to numpy.
trainYLabels = trainY[1].to_numpy()
trainYImages = trainY[0].to_numpy()
testYLabels = testY[1].to_numpy()
testYImages = testY[0].to_numpy()

# Start model learning.
learnInput = {}
learnInput['images'] = trainYImages
learnInput['features'] = trainX
learnInput['labels'] = trainYLabels
learnOutput = xDNNLearn(learnInput)

print ("Model Trained")
print("Model Validation...")

# Get the images, features and labels for validation phase
valInput = {}
valInput['xDNNParms'] = learnOutput['xDNNParms']
valInput['images'] = testYImages
valInput['features'] = testX
valInput['labels'] = testYLabels 

# Get and print results
valOutput = xDNNVal(valInput)
print('Accuracy: %f' % metrics.accuracy_score(testYLabels , valOutput['estLabel']))
print("Confusion Matrix: ")
print(confusion_matrix(testYLabels , valOutput['estLabel']))
