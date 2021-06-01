# Import the important modules
from keras.models import Model
from keras.models import save_model, load_model
from tensorflow.keras.utils import plot_model
from keras.layers import Input
from keras.layers import Conv2D, Dense, Cropping2D, concatenate
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import os
from matplotlib import pyplot as plt

from dir01_Moduldir import constants as con

# Important code to use a gpu with tf. May be removed depending on the device
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class NewNet():
    def __init__(self, path_load_model, path_save_model, path_train_images, path_test_images,
                 subdir_fency_pets, dir_original_pets, path_csv_loss, path_csv_valloss, path_predictions):

        # Path from which model is loaded. If it is empty a new model will be created.
        self.path_load_model = path_load_model

        # Path to save the model
        self.path_save_model = path_save_model

        # Path to train images
        self.path_train_images = path_train_images

        # Path to test images
        self.path_test_images = path_test_images

        # Subdirectory to images: pets with fences
        self.subdir_fency_pets = subdir_fency_pets

        # Subdirectory to images: original pets
        self.dir_original_pets = dir_original_pets

        # Path to save the csv file with trainings loss
        self.path_csv_loss = path_csv_loss

        # Path to save the csv file with validation loss
        self.path_csv_valloss = path_csv_valloss

        # Path to save predicted images
        self.path_predictions = path_predictions

    # Generate a dataset from two generators
    def image_dataset_generator(self, generator_name, first_image_generator, second_image_generator):
        generator_name = zip(first_image_generator, second_image_generator)
        for (first_image, second_image) in generator_name:
            yield first_image_generator.next(), second_image_generator.next()

    # Train the model
    def train_newnet(self):
        train_data_generator = ImageDataGenerator(rescale=1./255,
                                                  validation_split=con.val_split)

        # Generate a trainings data set from images with pet plus fence
        print("Creating trainings data set with fency images as x-values")
        image_data_generator = train_data_generator.flow_from_directory(
            self.path_train_images + self.subdir_fency_pets,
            target_size=(con.image_input_size, con.image_input_size),
            seed=con.seed,
            batch_size=con.trainings_batch_size,
            class_mode=None,
            color_mode="grayscale",
            subset='training'
        )

        # Generate a data set with original images
        print("Creating trainings data set with original images as y-values")
        mask_data_generator = train_data_generator.flow_from_directory(
            self.dir_original_pets,
            target_size=(con.image_output_size, con.image_output_size),
            seed=con.seed,
            batch_size=con.trainings_batch_size,
            class_mode=None,
            color_mode="grayscale",
            subset='training'
        )

        # Concatenate both datasets with one generator, to assign the right original image to each fency-pet-image
        training_generator = iter(())
        training_generator = self.image_dataset_generator(training_generator, image_data_generator, mask_data_generator)

        # Validation Data
        print("Creating validation data set with fency images as x-values")
        image_validation_generator = train_data_generator.flow_from_directory(
            self.path_train_images + self.subdir_fency_pets,
            target_size=(con.image_input_size, con.image_input_size),
            seed=con.seed,
            class_mode=None,
            color_mode="grayscale",
            subset='validation')

        print("Creating validation data set with original images as y-values")
        mask_validation_generator = train_data_generator.flow_from_directory(
            self.dir_original_pets,
            target_size=(con.image_output_size,con.image_output_size),
            seed=con.seed,
            class_mode=None,
            color_mode="grayscale",
            subset='validation')

        validation_generator = iter(())
        validation_generator = self.image_dataset_generator(
            validation_generator, image_validation_generator, mask_validation_generator)

        # Load an existing model or create a new one
        if os.path.isdir(self.path_load_model) is True:
            model = load_model(self.path_load_model)
            print("\nModel loaded from: " + str(self.path_load_model) + str("\n"))

        else:
            model = self.create_model()
            print("\nNew model created\n")

        # Configure checkpointer to save a trained model
        checkpointer = ModelCheckpoint(
            filepath=self.path_save_model,
            save_weights_only=False,
            monitor='val_loss',
            verbose=1,
            mode='min',
            save_best_only=True)

        # Train the model
        print("Train the model. Model will be saved at " + self.path_save_model)
        history = model.fit(
            training_generator,
            validation_data=validation_generator,
            validation_steps=con.val_steps,
            callbacks=[checkpointer],
            epochs=con.number_of_epochs,
            steps_per_epoch=con.steps_per_epoch
            )

        # create a CSV file for writing loss in the case the file does not already exist:
        if not os.path.exists(self.path_csv_loss):
            with open(self.path_csv_loss, "w"):
                print("New csv-file " + self.path_csv_loss + " created to save trainings loss")
                pass

        # write the loss to the csv file
        df = pd.DataFrame(data=history.history['loss'])
        df.to_csv(self.path_csv_loss, mode='a', index=False, header=False)
        print("Loss written to " + self.path_csv_loss)

        # create a CSV file for writing validation loss in the case the file does not already exist:
        if not os.path.exists(self.path_csv_valloss):
            with open(self.path_csv_valloss, "w"):
                print("New csv-file " + self.path_csv_loss + " created to save validaton loss")
                pass

        # write the loss to the csv file
        df = pd.DataFrame(data=history.history['val_loss'])
        df.to_csv(self.path_csv_valloss, mode='a', index=False, header=False)
        print("Validation loss written to " + self.path_csv_valloss)

    # Make predictions with a trained model
    def predict_with_newnet(self, number_of_predictions, show_predictions):
        # Create a test data set
        print("\nCreating data set from images")
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_image_data_generator = test_datagen.flow_from_directory(
            self.path_test_images + self.subdir_fency_pets,
            batch_size=con.prediction_batch_size,
            target_size=(con.image_input_size, con.image_input_size),
            seed=con.seed,
            shuffle=False,
            class_mode=None,
            color_mode="grayscale",
            subset='training')

        # Check how many images can be predicted
        test_data_filenames = test_image_data_generator.filenames
        max_predictions = len(test_data_filenames)

        if max_predictions < number_of_predictions:
            images_for_prediction = max_predictions
            print("Only " + str(images_for_prediction) + " are in your data set")
        else:
            images_for_prediction = number_of_predictions
            print(str(images_for_prediction) + " images will be predicted")

        # Load the model
        # If no path_load_model is given, try from path_save_model
        if os.path.isdir(self.path_load_model):
            model = load_model(self.path_load_model)
            print("Loaded model from " + self.path_load_model)
        else:
            model = load_model(self.path_save_model)
            print("Loaded model from " + self.path_save_model)

        # Predict images
        prediction = model.predict(test_image_data_generator, steps=images_for_prediction)

        # Save and optionally show the images
        for i in range(0, images_for_prediction):
            image_save_number = 1201 + i

            # save the predicted image
            path_save_prediction = self.path_predictions + "Prediction" + str(image_save_number) + '.jpg'
            plt.imsave(path_save_prediction, prediction[i, :, :, 0], cmap="gray")
            print("Prediction" + str(image_save_number) + ".jpg saved")

            # show the predicted image
            if i < show_predictions:
                plt.imshow(prediction[i, :, :, 0], cmap="gray")
                plt.show()

    # Create the model
    def create_model(self):
        # Contracting Path
        # Layerstack 1
        # Input size: 286
        input_shape = Input(shape=(con.image_input_size, con.image_input_size, 1))
        cp_s1_conv_layer1 = Conv2D(con.depth * 1, kernel_size=3, activation='relu', padding="same")(input_shape)
        cp_s1_conv_layer2 = Conv2D(con.depth * 1, kernel_size=3, activation='relu', padding="same")(cp_s1_conv_layer1)
        cp_s1_pool_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cp_s1_conv_layer2)

        # Layerstack 2
        cp_s2_conv_layer1 = Conv2D(con.depth * 2, kernel_size=3, activation='relu', padding="same")(cp_s1_pool_layer)
        cp_s2_conv_layer2 = Conv2D(con.depth * 2, kernel_size=3, activation='relu', padding="same")(cp_s2_conv_layer1)
        cp_s2_pool_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cp_s2_conv_layer2)

        # Layerstack 3
        cp_s3_conv_layer1 = Conv2D(con.depth * 4, kernel_size=3, activation='relu', padding="same")(cp_s2_pool_layer)
        cp_s3_conv_layer2 = Conv2D(con.depth * 4, kernel_size=3, activation='relu', padding="same")(cp_s3_conv_layer1)
        cp_s3_pool_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cp_s3_conv_layer2)

        # Layerstack 4
        cp_s4_conv_layer1 = Conv2D(con.depth * 8, kernel_size=3, activation='relu', padding="same")(cp_s3_pool_layer)
        cp_s4_conv_layer2 = Conv2D(con.depth * 8, kernel_size=3, activation='relu', padding="same")(cp_s4_conv_layer1)
        cp_s4_pool_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cp_s4_conv_layer2)

        # Middle Layer
        Mid_conv_layer1 = Conv2D(con.depth * 16, kernel_size=3, padding="same", activation='relu')(cp_s4_pool_layer)
        Mid_conv_layer2 = Conv2D(con.depth * 16, kernel_size=3, padding="same", activation='relu')(Mid_conv_layer1)

        # Expanding Path
        ep_s4_transp_layer = Conv2DTranspose(
            64, kernel_size=2, activation='relu', strides=(2, 2), padding="same")(Mid_conv_layer2)
        ep_s4_crop_layer = Cropping2D(cropping=((1, 0), (1, 0)))(cp_s4_conv_layer2)   # 34
        ep_s4_conc_layer = concatenate([ep_s4_crop_layer, ep_s4_transp_layer])
        ep_s4_conv_layer2 = Conv2D(con.depth * 8, kernel_size=(3, 3), padding="same", activation="relu")\
            (ep_s4_conc_layer)
        ep_s4_conv_layer1 = Conv2D(con.depth * 8, kernel_size=(3, 3), padding="same", activation="relu")\
            (ep_s4_conv_layer2)

        # Layerstack 3
        ep_s3_transp_layer = Conv2DTranspose(
            64, kernel_size=2, activation='relu', strides=(2, 2), padding="same")(ep_s4_conv_layer1)
        ep_s3_crop_layer = Cropping2D(cropping=((2, 1), (2, 1)))(cp_s3_conv_layer2)  # 68
        ep_s3_conc_layer = concatenate([ep_s3_crop_layer, ep_s3_transp_layer])
        ep_s3_conv_layer2 = Conv2D(con.depth * 4, kernel_size=(3, 3), padding="same", activation="relu")\
            (ep_s3_conc_layer)
        ep_s3_conv_layer1 = Conv2D(con.depth * 4, kernel_size=(3, 3), padding="same", activation="relu")\
            (ep_s3_conv_layer2)

        # Layerstack 2
        ep_s2_transp_layer = Conv2DTranspose(
            64, kernel_size=2, activation='relu', strides=(2, 2), padding="same")(ep_s3_conv_layer1)
        ep_s2_crop_layer = Cropping2D(cropping=((4, 3), (4, 3)))(cp_s2_conv_layer2)  # 136
        ep_s2_conc_layer = concatenate([ep_s2_crop_layer, ep_s2_transp_layer])
        ep_s2_conv_layer2 = Conv2D(con.depth * 2, kernel_size=(3, 3), padding="same", activation="relu")\
            (ep_s2_conc_layer)
        ep_s2_conv_layer1 = Conv2D(con.depth * 2, kernel_size=(3, 3), padding="same", activation="relu")\
            (ep_s2_conv_layer2)

        # Layerstack 1
        ep_s1_transp_layer = Conv2DTranspose(
            64, kernel_size=2, activation='relu', strides=(2, 2), padding="same")(ep_s2_conv_layer1)
        ep_s1_crop_layer = Cropping2D(cropping=((7, 7), (7, 7)))(cp_s1_conv_layer2) # 272
        ep_s1_conc_layer = concatenate([ep_s1_crop_layer, ep_s1_transp_layer])
        ep_s1_conv_layer2 = Conv2D(con.depth * 1, kernel_size=(3, 3), padding="same", activation="relu")\
            (ep_s1_conc_layer)
        ep_s1_conv_layer1 = Conv2D(con.depth * 1, kernel_size=(3, 3), padding="same", activation="relu")\
            (ep_s1_conv_layer2)

        # Output Layer
        ep_s1_out_layer = Conv2D(1, kernel_size=1, padding="same", activation='sigmoid')(ep_s1_conv_layer1)

        # Build the Model
        model = Model(inputs=input_shape, outputs=ep_s1_out_layer)
        plot_model(model, to_file='NewNet.png', show_shapes=True, show_layer_names=True)

        # compile model
        adam = Adam(lr=con.lernrate)
        model.compile(optimizer=adam, loss="mse")
        return model
