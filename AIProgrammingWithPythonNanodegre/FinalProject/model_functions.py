import torch
import time

from collections import OrderedDict
from torch import nn, optim
from torchvision import models
from utility_functions import process_image


def create_new_model(arch, hidden_units):
    '''Function loads new pretrained CNN model (tourch.modules), freezes its parameters
    and creates new classifier. Available architectures: vgg1, vgg19, densenet121, alexnet
    '''
    # List of supported achitectures
    in_features = {
        'vgg11': 25088,
        'vgg19': 25088,
        'densenet121': 1024,
        'alexnet': 9216
    }

    # Create new model and freeze params
    model = getattr(models, arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # Create new classirier according to model architecture
    model.classifier = create_new_classifier(in_features[arch], hidden_units)

    return model


def create_new_classifier(in_features, hidden_units, out_features=102):
    '''Function creates new classifier for CNN.
    '''
    classifier = nn.Sequential(OrderedDict([('input', nn.Linear(in_features=in_features, out_features=hidden_units, bias=True)),
                                            ('relu', nn.ReLU()),
                                            ('droput', nn.Dropout(0.2)),
                                            ('h1', nn.Linear(in_features=hidden_units, out_features=out_features, bias=True)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    return classifier


def train_model(model, lr, epochs, train_data, valid_data, use_gpu):
    '''Function trains model accroding to parameters. Function uses NLLLoss function as a criterion
    and Adam optimizer.
    '''
    # Checking if GPU available
    if torch.cuda.is_available() and use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    for e in range(epochs):

        start_time = time.time()
        train_loss = 0

        for inputs, labels in train_data:

            # set gradient to zero and enable dropout
            model.train()
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)

            # get log probabilities from output and calculate loss
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # backpropagation and gradient descent
            loss.backward()
            optimizer.step()

            train_loss += loss

        # disable calculating grad to test model and disable droput 
        with torch.no_grad():

            model.eval()
            valid_loss = 0
            accuracy = 0

            for inputs, labels in valid_data:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                valid_loss += loss

                # calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        end_time = time.time() - start_time

        print(f"Epoch {e + 1:02}/{epochs} | "
              f"Train loss = {train_loss/len(train_data):.3f} | "
              f"Validation loss = {valid_loss/len(valid_data):.3f} | "
              f"Validation accuracy = {accuracy/len(valid_data):.3f} | "
              f"Time: {end_time//60:.0f}m {end_time % 60:.0f}s")


def save_model_checkpoint(model, arch, save_dir):
    '''Function saves model parameters.
    '''
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'arch': arch,
                  'class_to_idx': model.class_to_idx
                  }

    torch.save(checkpoint, save_dir + '/checkpoint_by_cmd.pth')


def load_model_checkpoint(path_to_checkpoint):
    '''Function returns a model with parameters from checkpoint
    '''
    checkpoint = torch.load(path_to_checkpoint, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    return model


def predict_classes(image, model, topk, use_gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if torch.cuda.is_available() and use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)
    image = image.to(device)

    # get prediction and top K probs
    ps = torch.exp(model.forward(image))
    probs, classes = ps.topk(topk)

    # transform probs and classes to list
    probs = probs.view(topk).tolist()
    classes = classes.view(topk).tolist()

    keys = list(model.class_to_idx.keys())

    for i in range(topk):
        classes[i] = keys[classes[i]]

    return probs, classes
