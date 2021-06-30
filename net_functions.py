from sys import dont_write_bytecode
import torch
from torch import nn
from torch.nn import functional as f
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import BatchSampler

def GenerateDatasets(trainDS, testDS, show=False):
    #Splitting the training datasets into 90% training data and 10% validation data
    training_dataset, val_dataset = random_split(trainDS, (45000, 5000))
    #Generation of dataloaders and batching packages for each of the training, validation and testing datasets
    trainLoader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    valLoader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    testLoader = DataLoader(testDS, batch_size=64, shuffle=True)
    #Reporting of the generated dataloaders and bacthes
    if show:
        print(f"Total batches created in the train loader: {len(trainLoader)}.")
        print(f"Total batches created in the validation loader: {len(valLoader)}.")
        print(f"Total batches created in the test loader: {len(testLoader)}.")
        print(f"Size of the training dataset is: {len(trainLoader.dataset)}.")
        print(f"Size of the validation dataset is: {len(valLoader.dataset)}.")
        print(f"Size of the testing dataset is: {len(testLoader.dataset)}.")
        
    return trainLoader, valLoader, testLoader

def multiclass_accuracy(y_pred, y_true):
    top_p, top_class = y_pred.topk(1,dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

def train_model_epochs(model, trainLoader, valLoader, epochs=10, lr=1e-3):
    criterion = nn.NLLLoss() #Logaritmic probabilities versus 'true' labels loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #Training repetition by epochs
    for i in range(epochs):
        train_loss = 0.
        train_acc = 0.
        valid_loss = 0.
        valid_acc = 0.
        #Model training stage
        model.train()
        for images, labels in trainLoader:
            logps = model(images)
            loss = criterion(logps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            ps = torch.exp(logps)
            train_acc += multiclass_accuracy(ps, labels)
        #Average loss and accuracy calculations
        avg_train_loss = train_loss/len(trainLoader)
        avg_train_acc = train_acc/len(trainLoader)
        print(f"epoch: {i} - Training Loss: {avg_train_loss:.4f} - Training Accuracy: {avg_train_acc:.4f}")
        #Model validation stage
        model.eval()
        for images, labels in valLoader:
            logps = model(images)
            loss = criterion(logps, labels)
            valid_loss += loss.item()
            ps = torch.exp(logps)
            valid_acc += multiclass_accuracy(ps, labels)
        #Average loss and accuracy calculations
        avg_valid_loss = valid_loss/len(valLoader)
        avg_valid_acc = valid_acc/len(valLoader)
        print(f"epoch: {i} - Validation Loss: {avg_valid_loss:.4f} - Validation Accuracy: {avg_valid_acc:.4f}")

def train_model_acc(model, trainLoader, valLoader, accTarget=0.95, lr=1e-3):
    criterion = nn.NLLLoss() #Logaritmic probabilities versus 'true' labels loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    valid_acc = 0
    i = 0
    while(accTarget > valid_acc):
        i += 1 
        train_loss = 0.
        train_acc = 0.
        valid_loss = 0.
        valid_acc = 0.
        #Model training stage
        model.train()
        for images, labels in trainLoader:
            logps = model(images)
            loss = criterion(logps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            ps = torch.exp(logps)
            train_acc += multiclass_accuracy(ps, labels)
        #Average loss and accuracy calculations
        avg_train_loss = train_loss/len(trainLoader)
        avg_train_acc = train_acc/len(trainLoader)
        print(f"epoch: {i} - Training Loss: {avg_train_loss:.4f} - Training Accuracy: {avg_train_acc:.4f}")
        #Model validation stage
        model.eval()
        for images, labels in valLoader:
            logps = model(images)
            loss = criterion(logps, labels)
            valid_loss += loss.item()
            ps = torch.exp(logps)
            valid_acc += multiclass_accuracy(ps, labels)
        #Average loss and accuracy calculations
        avg_valid_loss = valid_loss/len(valLoader)
        avg_valid_acc = valid_acc/len(valLoader)
        print(f"epoch: {i} - Validation Loss: {avg_valid_loss:.4f} - Validation Accuracy: {avg_valid_acc:.4f}")

def test_model(model, testLoader):
    criterion = nn.NLLLoss() #Logaritmic probabilities versus 'true' labels loss
    test_loss = 0.
    test_acc = 0.

    model.eval()
    for images, labels in testLoader:
        logps = model(images)
        loss = criterion(logps, labels)
        test_loss += loss.item()
        ps = torch.exp(logps)
        test_acc += multiclass_accuracy(ps, labels)
    
    avg_test_loss = test_loss/len(testLoader)
    avg_test_acc = test_acc/len(testLoader)
    print(f"Test Loss: {avg_test_loss:.4f} - Test Accuracy: {avg_test_acc:.4f}")
    
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        #Convolutional layers in 2D
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), padding=1, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1, stride=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, stride=1)
        #Maxpooling function in 2D for convolution
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #Linear layers in 1D
        self.linear_1 = nn.Linear(in_features=1024, out_features=500)
        self.linear_2 = nn.Linear(in_features=500, out_features=128)
        self.linear_3 = nn.Linear(in_features=128, out_features=10)
    
    def forward(self, images):
        a1 = self.maxpool(f.relu(self.conv_1(images)))
        a2 = self.maxpool(f.relu(self.conv_2(a1)))
        a3 = self.maxpool(f.relu(self.conv_3(a2)))
        a3 = a3.view(a3.shape[0], -1) #Tensor shape (64, 4, 4) => Tensor shape (1024)
        a4 = f.relu(self.linear_1(a3))
        a5 = f.relu(self.linear_2(a4))
        return f.log_softmax(self.linear_3(a5), dim=1)