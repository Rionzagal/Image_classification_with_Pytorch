import os
from net_functions import *
from utils import *

##Loading CIFAR-10 Dataset
from torchvision import transforms as T, datasets
data_transforms = T.Compose([
    T.ToTensor(), 
    T.Normalize(mean =[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]) #See README file for more information
    ])

if not os.path.exists('./cifar10/'): os.makedirs('./cifar10/')
train_ds = datasets.CIFAR10('cifar10/', train=True, download=~os.path.exists('./cifar10/'), transform=data_transforms)
test_ds = datasets.CIFAR10('cifar10/', train=False, download=~os.path.exists('./cifar10/'), transform=data_transforms)

print(f"Train set has a size of: {len(train_ds)} tensors.")
print(f"Test set has a size of {len(test_ds)} tensors.")

trainLoader, valLoader, testLoader = GenerateDatasets(train_ds, test_ds, show=True)

net = ConvModel()

train_model_epochs(model=net, trainLoader=trainLoader, valLoader=valLoader, epochs=10, lr=1e-3)

test_model(model=net, testLoader=testLoader)

dataIter = iter(testLoader)
images, labels = dataIter.next()
index = np.random.randint(low=0, high=63, dtype=np.uint8)
logps = net(images[index].unsqueeze(0))
ps = torch.exp(logps)

view_classify(images[index], ps)