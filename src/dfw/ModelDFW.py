import torch
import torch.nn as nn
from torch.nn import functional as F

from src.dfw import DFW
from src.dfw.losses import MultiClassHingeLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ThreeLayer(nn.Module):
    def __init__(self, d0, d1, d2, d3, classes):
        super(ThreeLayer, self).__init__()

        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.classes = classes

        self.fc1 = nn.Linear(d0, d1).to(device=device)
        self.fc2 = nn.Linear(d1, d2).to(device=device)
        self.fc3 = nn.Linear(d2, d3).to(device=device)
        self.fc4 = nn.Linear(d3, classes).to(device=device)

    def forward(self, x):
        x = x.to(device=device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)

        return out


class ModelDFW:

    def __init__(self, d0, d1, d2, d3, classes):
        self.model = ThreeLayer(d0, d1, d2, d3, classes)

    def train(self, train_data, test_data, n_epochs, lr=1e-1, verbose=True):

        # Create DFW optimizer
        optimizer = DFW(self.model.parameters(), eta=lr)
        # Instantiate loss
        criterion = MultiClassHingeLoss()

        tr_acc = []
        te_acc = []

        test = []
        train_size = len(train_data)
        test_size = len(test_data)
        ratio = int(train_size / test_size) - 1

        # Puts ratio*(None, None) between each test batch so that test array is the same length as train_data
        if ratio > 0:
            for i, j in enumerate(test_data):
                test.append(j)
                for r in range(ratio):
                    test.append((None, None))
        else:
            test = test_data

        for e in range(n_epochs):

            epoch_tr_loss = 0.0
            epoch_te_loss = 0.0
            epoch_te_acc = 0.0
            epoch_tr_acc = 0.0

            for (tr_inputs, tr_classes), (te_inputs, te_classes) in zip(train_data, test):

                tr_inputs = tr_inputs.to(device=device)
                tr_classes = tr_classes.to(device=device)

                # Flatten the train input to feed it to the fully connected network
                tr_inputs = tr_inputs.view(-1, tr_inputs.shape[1] * tr_inputs.shape[2] * tr_inputs.shape[3])

                # Forward pass
                tr_output = self.model(tr_inputs)

                # Compute the loss for each batch
                tr_loss = criterion(tr_output, tr_classes.long())

                # Sum all the losses to get the epoch's loss
                epoch_tr_loss += tr_loss.item()

                # Compute the accuracy for each batch and sum them
                epoch_tr_acc += self.test_batch(tr_inputs, tr_classes)

                if te_inputs != None:
                    te_inputs = te_inputs.to(device=device)
                    te_classes = te_classes.to(device=device)

                    # Same as train inputs
                    te_inputs = te_inputs.view(-1, te_inputs.shape[1] * te_inputs.shape[2] * te_inputs.shape[3])
                    te_output = self.model(te_inputs)
                    te_loss = criterion(te_output, te_classes.long())
                    epoch_te_loss += te_loss.item()
                    epoch_te_acc += self.test_batch(te_inputs, te_classes)

                # Backward pass
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step(lambda: float(tr_loss))

            # Compute the epoch losses and accuracies by averaging the summed values
            epoch_tr_loss = epoch_tr_loss / train_size
            epoch_te_loss = epoch_te_loss / test_size
            epoch_tr_acc = epoch_tr_acc / train_size
            epoch_te_acc = epoch_te_acc / test_size

            # Keep those values to plot them and get insights about the training
            tr_acc.append(epoch_tr_acc)
            te_acc.append(epoch_te_acc)

            if verbose:
                print(
                    f"Epoch: {e + 1} / {n_epochs} \n Train loss: {epoch_tr_loss:.4f} - Test loss:{epoch_te_loss:.4f} \n Train acc: {tr_acc[e]:.4f} - Test acc: {te_acc[e]:.4f}")

        return tr_acc, te_acc

    # Computes the accuracy where the "test_data" is a batch (already flattened)
    def test_batch(self, test_data, test_classes):

        # Init the number of correct predictions
        nb_correct = 0

        # Number of samples
        N = len(test_classes)

        # Compute predicted labels by the model
        model_output = self.model(test_data)

        # Get the targets
        predicted_labels = torch.argmax(model_output, 1, keepdim=True).view(test_classes.size()[0])

        # Count the number of correct predictions
        nb_correct += (predicted_labels == test_classes).int().sum().item()

        return nb_correct / N

    # Computes the accuarcy where test_data is a DataLoader
    def test(self, test_data, batch_size):
        """Test method using the error rate as a metric """
        # Init the number of correct predictions
        nb_correct = 0

        # Number of samples
        N = len(test_data) * batch_size

        # Iterate over each batch of the Loader
        for images, labels in iter(test_data):
            images = images.to(device=device)
            labels = labels.to(device=device)

            # Flatten each batch to feed it to our model
            images = images.view(-1, images.shape[1] * images.shape[2] * images.shape[3])

            # Run the model on a mini batch of the images
            model_output = self.model(images)

            # Get the targets
            predicted_labels = torch.argmax(model_output, 1, keepdim=True).view(labels.size()[0])

            # Count the number of correct predictions
            nb_correct += (predicted_labels == labels).int().sum().item()

        return nb_correct / N
