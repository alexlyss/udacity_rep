# PURPOSE: Program loads pretrained model for image classification 
#          and uses it as classifier for flower species. Flower labels are taken from 
#          JSON file according to train dataset
#          Input parameters: 
#                           - path to image
#                           - path to pretrained model
#
#   Example call:
#           python predict.py /home/images/image_07833.jpg  /home/checkpoint.pth --category_names ./cat_to_name.json


import logging

from model_functions import *
from utility_functions import *


def main():

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    # parse arguments
    arguments = get_args_for_predict()

    # load model from checkpoint
    logging.info(f'Loading model from checkpoint file {arguments.checkpoint}')
    my_model = load_model_checkpoint(arguments.checkpoint)

    # prepare tensor for model
    logging.info(f'Preprocessing the image {arguments.input}')
    image = process_image(arguments.input)

    # predict classes and their porbabilities
    logging.info('Getting classes and probabilities')
    probs, classes = predict_classes(image, my_model, arguments.top_k, arguments.use_gpu)

    # transform classes to readable format according to category_names
    if arguments.category_names:
        try:
            classes = load_category_names(classes, arguments.category_names)
        except Exception as e:
            logging.warning(f'Cannot get category names {e}')

    # Show results
    print('\n| {:>25} | {:<14}|'.format('Flower species', 'Probabilites'))
    for prob, class_name in zip(probs, classes):
        print(f'| {class_name:>25} | {prob:<14.2%}|')

if __name__ == "__main__":
    main()
