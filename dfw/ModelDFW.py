import torch
import torch.nn as nn
from dfw.losses import MultiClassHingeLoss
from dfw import DFW
from torch.nn import functional as F


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
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        
        return out
    
    
def three_layer(d0, d1, d2, d3, classes):
    model = ThreeLayer(d0, d1, d2, d3, classes)
    return model

class ModelDFW:
    
    def __init__(self, d0, d1, d2, d3, classes):
        self.model = three_layer(d0, d1, d2, d3, classes)
        
        
    def train(self, train_data, test_data, n_epochs, lr=1e-1, verbose=False):
        
        # create DFW optimizer 
        optimizer = DFW(self.model.parameters(), eta=lr)
        criterion = MultiClassHingeLoss()
        
        tr_losses = []
        te_losses = []
        tr_acc = []
        te_acc = []
        
        for e in range(n_epochs):
            epoch_tr_loss = 0.0
            epoch_te_loss = 0.0
            epoch_te_acc = 0.0
            epoch_tr_acc = 0.0
            #Pas trop de sens de test comme ca mais similaire à BCD    
            for (tr_inputs, tr_classes), (te_inputs, te_classes)  in zip(train_data, test_data):
                #tr_inputs = next(iter(train_data))[0]
                #tr_classes = next(iter(train_data))[1]

                #te_inputs = next(iter(test_data))[0]
                #te_classes = next(iter(test_data))[1]
                
                tr_inputs = tr_inputs.view(-1, tr_inputs.shape[1] * tr_inputs.shape[2] * tr_inputs.shape[3])
                te_inputs = te_inputs.view(-1, te_inputs.shape[1] * te_inputs.shape[2] * te_inputs.shape[3])

                # Forward pass
                tr_output = self.model(tr_inputs)
                te_output = self.model(te_inputs)

                tr_loss = criterion(tr_output, tr_classes.long())
                te_loss = criterion(te_output, te_classes.long())

                # Apply the backward step
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step(lambda: float(tr_loss))

                #tr_losses.append(tr_loss)
                #te_losses.append(te_loss)
                epoch_tr_loss += tr_loss.item()
                epoch_te_loss += te_loss.item()
                epoch_tr_acc += self.test_batch(tr_inputs, tr_classes)
                epoch_te_acc += self.test_batch(te_inputs, te_classes)
                
                #tr_acc.append(self.test(tr_inputs, tr_classes))
                #te_acc.append(self.test(te_inputs, te_classes))
            epoch_tr_loss = epoch_tr_loss / len(train_data)
            epoch_te_loss = epoch_te_loss / len(test_data)
            epoch_tr_acc = epoch_tr_acc / len(train_data)
            epoch_te_acc = epoch_te_acc / len(test_data)
            tr_losses.append(epoch_tr_loss)
            te_losses.append(epoch_te_loss)
            tr_acc.append(epoch_tr_acc)
            te_acc.append(epoch_te_acc)
            if verbose:
                print(f"Epoch: {e + 1} / {n_epochs} \n Train loss: {tr_losses[e]:.4f} - Test loss:{te_losses[e]:.4f} \n Train acc: {tr_acc[e]:.4f} - Test acc: {te_acc[e]:.4f}")
                
        return tr_losses, te_losses, tr_acc, te_acc
        

    def test_batch(self, test_data, test_classes):
       
        # Init the number of correct predictions
        nb_correct = 0
                
        # Number of samples
        N = len(test_classes)

        model_output = self.model(test_data)
                
            # Get the targets
        predicted_labels = torch.argmax(model_output,1,keepdim=True).view(test_classes.size()[0])
         
                
            # Count the number of correct predictions
        nb_correct +=(predicted_labels == test_classes).int().sum().item()
            
        return nb_correct / N
       
    def test(self, test_data, batch_size):
        """Test method using the error rate as a metric """
        # Init the number of correct predictions
        nb_correct = 0
                
        # Number of samples
        N = len(test_data) * batch_size 

        for images, labels in iter(test_data):
            
            images = images.view(-1, images.shape[1] * images.shape[2] * images.shape[3])
            # Run the model on a mini batch of the images
            model_output = self.model(images)
                
            # Get the targets
            predicted_labels = torch.argmax(model_output,1,keepdim=True).view(labels.size()[0])
                
            # Count the number of correct predictions
            nb_correct +=(predicted_labels == labels).int().sum().item()
            
        return nb_correct / N
       