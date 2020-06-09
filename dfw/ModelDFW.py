import torch
import torch.nn as nn
from losses import MultiClassHingeLoss


class ThreeLayer(nn.Module):
    def __init__(self, d0, d1, d2, d3, classes):
        super(ThreeLayer, self).__init__()
        
        
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.classes = classes

        self.fc1 = nn.Linear(d0, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.fc3 = nn.Linear(d2, d3)
        self.fc4 = nn.Linear(d3, classes)
        
       
        
    def forward(self, x):
        
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = nn.ReLU(self.fc3(x))
        out = self.fc4(x)
        
        return out
    
    
def three_layer(d0, d1, d2, d3, classes):
    model = ThreeLayer(d0, d1, d2, d3, classes)
    return model

class ModelDFW:
    
    def __init__(self, d0, d1, d2, d3, classes):
        self.model = three_layer(d0, d1, d2, d3, classes)
        
        
    def train(self, train_data, test_data, nb_epochs=50, lr=1e-1, verbose=True):
        
        # create DFW optimizer 
        optimizer = DFW(self.model.parameters(), eta=lr)
        criterion = MultiClassHingeLoss()
        
        tr_losses = []
        te_losses = []
        tr_acc = []
        te_acc = []
        
        for e in range(nb_epochs):
            if verbose and e  != 0:
                print("Epochs {}".format(e))
                print("loss = {}".format(loss))
            
            for (tr_inputs, tr_classes), (te_inputs, te_classes) in zip(train_loader, test_loader):
                
                tr_inputs = tr_inputs.view(-1, tr_inputs.shape[1] * tr_inputs.shape[2] * tr_inputs.shape[3])
                te_inputs = te_inputs.view(-1, te_inputs.shape[1] * te_inputs.shape[2] * te_inputs.shape[3])
                
                # Forward pass
                tr_output = self.model(tr_inputs)
                te_output = self.model(te_inputs)
                
                tr_loss = criterion(tr_output, tr_classes.long())
                te_loss = criterion(te_output, te_classes.long())
                
                tr_losses.append(tr_loss)
                te_losses.append(te_loss)
                
                
                # Apply the backward step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(lambda: float(loss))
        

    def test(self, test_data):
       
        # Init the number of correct predictions
        nb_correct = 0
                
        # Number of samples
        N = len(test_data)

        for images, labels in iter(test_data):
            # Run the model on a mini batch of the images
            model_output = self.model(images)
                
            # Get the targets
            predicted_labels = torch.argmax(model_output,1,keepdim=True).view(labels.size()[0])
                
            # Count the number of correct predictions
            nb_correct +=(predicted_labels == labels).int().sum().item()
            
        return nb_correct / N
       