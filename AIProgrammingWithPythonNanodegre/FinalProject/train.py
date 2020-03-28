# PURPOSE: Program trains and saves model using one of the CNN architecture.
#          The model trains for the classification of flower species on the images.
#          Input parameters: 
#                           - path to dataset folder
#                           - path to directory where program saves model
#
#   Example call:
#    python train.py ./flowers --gpu --save_dir /home/checkpoints


# Import modules
import logging

from model_functions import *
from utility_functions import *


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Parse input arguments
    arguments = get_args_for_train()

    # Prepare data sets 
    logging.info('Prepairing datasets')
    train_dataset = get_dataset(data_directory=arguments.data_directory + '/train', is_train=True)
    valid_dataset = get_dataset(data_directory=arguments.data_directory + '/valid')

    # Get date loaders
    logging.info('Creating dataloaders')
    train_dataloader = get_data_loader(train_dataset, shuffle=True)
    valid_dataloader = get_data_loader(valid_dataset)

    # Create Model
    logging.info('Creating model')
    my_model = create_new_model(arguments.arch, arguments.hidden_units)
    my_model.class_to_idx = train_dataset.class_to_idx

    # Train Model
    logging.info('Training model')
    train_model(model=my_model,
                lr=arguments.learning_rate,
                epochs=arguments.epochs,
                train_data=train_dataloader,
                valid_data=valid_dataloader,
                use_gpu=arguments.use_gpu)

    # Save model
    logging.info(f'Saving model checkpoint to {arguments.save_dir} as checkpoint_by_cmd.pth' )
    save_model_checkpoint(my_model, arguments.arch, arguments.save_dir)


if __name__ == "__main__":
    main()
