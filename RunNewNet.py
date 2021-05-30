import argparse

from dir01_Moduldir import NewNet
import settings as set


def parse_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-t', '--train', type=bool, default=False)
    my_parser.add_argument('-l', '--path_load_model', type=str, default=set.path_load_model)
    my_parser.add_argument('-s', '--path_save_model', type=str, default=set.path_save_model)
    my_parser.add_argument('-n', '--number_of_predictions', type=int, default=set.number_of_predictions)
    my_parser.add_argument('-i', '--show_predictions', type=int, default=set.show_predictions)
    return vars(my_parser.parse_args())


args = parse_args()
model = NewNet.NewNet(args['path_load_model'],
                      args['path_save_model'],
                      set.path_train_images,
                      set.path_test_images,
                      set.subdir_fency_pets,
                      set.dir_original_pets,
                      set.path_csv_loss,
                      set.path_csv_valloss,
                      set.path_predictions
                      )

# Train a model based on settings.py
if args['train']:
    model.train_newnet()

# Predict images based on settings.py
model.predict_with_newnet(args['number_of_predictions'],
                          args['show_predictions'])
