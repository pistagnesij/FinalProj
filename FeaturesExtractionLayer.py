import pandas
import numpy
import os
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from sklearn.model_selection import train_test_split

 
#Class definition for Feature extraction 
class featuresEncoding(torch.nn.Module):
    def __init__(self):
        super(featuresEncoding, self).__init__()
        #Create pre-trained VGG-16 model
        vgg16 = torchvision.models.vgg16(pretrained=True)
        
        #Extract features using VGG weights
        self.features = vgg16.features 
        self.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        pretrained_dict = vgg16.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
 
    def forward(self, x):
        return self.classifier(self.features(x).view(self.features(x).size(0), -1))


#Load the data directory  where the images are stored
pics = 'pics/'
contents = os.listdir(pics)
classes = [each for each in contents if os.path.isdir(pics + each)]

cIterator = 0
labels = []
images = []
batch = []

model = featuresEncoding()

for each in classes:
  classPath = pics + each  
  print("{} images".format(each))
  files = os.listdir(classPath)
  
  for x, file in enumerate(files, 1):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )
    img = os.path.join(classPath, file)
    img = transform(Image.open(img))[:3]
    
    # Extract features using the VGG-16 structure
    features = torch.squeeze(model(Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)).cpu()).data.numpy()
    
    batch.append(features)
    images.append(file)
    labels.append(str(cIterator))
    print("finish {}".format(x))
  cIterator = cIterator + 1  

labelsNP = numpy.array(labels).reshape(-1,1)
imagesNP = numpy.array(images).reshape(-1,1)


# Make training/testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(numpy.array(batch), numpy.hstack((imagesNP,labelsNP)), test_size=0.3, random_state=0)

# Convert to pandas and save as CSV
trainX = pandas.DataFrame(Xtrain)
trainY = pandas.DataFrame(Ytrain)
testX = pandas.DataFrame(Xtest)
testY = pandas.DataFrame(Ytest)

print("Data converted and saved")
trainX.to_csv('trainX.csv',header=False,index=False)
trainY.to_csv('trainY.csv',header=False,index=False)
testX.to_csv('testX.csv',header=False,index=False)
testY.to_csv('testY.csv',header=False,index=False)
