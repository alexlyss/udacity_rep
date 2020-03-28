import argparse
import json
import numpy as np
import torch

from PIL import Image
from torchvision import transforms, datasets


def load_category_names(classes, path):
    '''Function loads labels for classes from JSON file. 
    '''
    with open(path, 'r') as f:
        category_names = json.load(f)

    class_names = [category_names[img_class] for img_class in classes]

    return class_names


def get_args_for_train():
    '''Function parses arguments for train.py 
    '''
    parser = argparse.ArgumentParser(description='Program for training NN for flower picture recognition')
    # path to data dir (mandatory)
    parser.add_argument('data_directory',
                        type=str,
                        help='Directory with dataset for training')

    # path to save checkpoint model
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        type=str,
                        action='store',
                        default='./',
                        help='Directory where model\'s checkpoint is saved after training. Default is current directory')

    # model architecture
    parser.add_argument('--arch',
                        dest='arch',
                        type=str,
                        choices=['vgg11', 'vgg19', 'densenet121', 'alexnet'],
                        action='store',
                        default='vgg11',
                        help='CNN architecture. Default is vgg11')

    # learning rate
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        type=float,
                        action='store',
                        default=0.001,
                        help='Learning coeficient for Adam. Default is 0.001')

    # hidden units
    parser.add_argument('--hidden_units',
                        dest='hidden_units',
                        type=int,
                        action='store',
                        default=512,
                        help='Number of hidden units in classifier. Default is 512')

    # epoches
    parser.add_argument('--epochs',
                        dest='epochs',
                        type=int,
                        action='store',
                        default=10,
                        help='Number of training epochs. Default is 10')

    # Use gpu gor trining
    parser.add_argument('--gpu',
                        dest='use_gpu',
                        action="store_true",
                        default=False,
                        help='Allows to use GPU for trainig. If there is no GPU will work with CPU')

    return parser.parse_args()


def get_args_for_predict():
    '''Function parses arguments for predict.py 
    '''
    parser = argparse.ArgumentParser(description='Program for prediction flower species')
    # path to data dir (mandatory)
    parser.add_argument('input',
                        type=str,
                        help='Image file path')

    parser.add_argument('checkpoint',
                        type=str,
                        help='Model\'s checlpoint file path')

    # path to save checkpoint model
    parser.add_argument('--top_k',
                        dest='top_k',
                        type=int,
                        action='store',
                        default=1,
                        help='Return top K most likely classes. Default is 3')

    # path to JSON file with category names
    parser.add_argument('--category_names',
                        dest='category_names',
                        type=str,
                        help='Path to JSON file with catogry names')

    # Use gpu gor trining
    parser.add_argument('--gpu',
                        dest='use_gpu',
                        action="store_true",
                        default=False,
                        help='Allows to use GPU. If there is no GPU will work with CPU')

    return parser.parse_args()


def get_data_transform(is_train):
    '''Function returns collection of datatransforms for images. 
    '''
    transforms_list = []
    transforms_list.append(transforms.Resize(255))

    if is_train:
        transforms_list.extend([transforms.RandomRotation(45),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip()]
                               )

    transforms_list.extend([transforms.RandomCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                            ])

    return transforms.Compose(transforms_list)


def get_dataset(data_directory, is_train=False):
    '''Function returns a dataset with torchision format from a folder 
    '''
    transform = get_data_transform(is_train)
    return datasets.ImageFolder(data_directory, transform=transform)


def get_data_loader(dataset, batch_size=32, shuffle=False):
    '''Function returns data loader for specified dataset 
    '''
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
    '''

    # define means and stds to normilize image
    n_means = np.array([0.485, 0.456, 0.406])
    n_stds = np.array([0.229, 0.224, 0.225])

    # open and resize image
    img = Image.open(image)
    w, h = img.size
    w, h = (256, round(256 * h/w)) if w < h else (round(256 * w/h), 256)
    img.thumbnail((w, h))
    img = img.crop(((w - 224)/2, (h - 224)/2, (w + 224)/2, (h + 224)/2))

    # represen image as numpy array and normilize
    np_image = np.array(img)
    np_image = np_image / 255
    np_image = (np_image - n_means)/n_stds
    np_image = np_image.transpose(2, 0, 1)

    # convert numpy array to torch tensor
    torch_image = torch.from_numpy(np_image)
    torch_image.unsqueeze_(0)
    torch_image = torch_image.float()
    return torch_image
