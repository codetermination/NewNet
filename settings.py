import datetime
import os


# Change trainings mode here
mode = "A"  # A = "NoBlur", B = "SingleBlur", C = "DoubleBlur"

# The model will be saved based on the time when the training starts
now = datetime.datetime.now()
minute = now.minute
hour = now.hour
day = now.day
month = now.month
year = now.year

# The name of the model is chosen from the mode
if mode == "A":
    modelmode = "NoBlur"
elif mode == "B":
    modelmode = "SingleBlur"
elif mode == "C":
    modelmode = "DoubleBlur"
else:
    print("Your model cannot be saved because you did not choose a valid model mode in settings")
    modelmode = ""

# For predictions only path_load_model is needed
path_load_model = "dir03_Models/NoBlur_2021_3_28_21_30/"  # /dir03_Models/<modelname>/
path_save_model = "dir03_Models/{0}_{1}_{2}_{3}_{4}_{5}".format(modelmode, year, month, day, hour, minute)
path_train_images = "dir02_Dataset/dir{0}_{1}/Train/".format(mode, modelmode)
path_test_images = "dir02_Dataset/dir{0}_{1}/Test/".format(mode, modelmode)
subdir_fency_pets = "FencyImgs/"
dir_original_pets = "dir02_Dataset/dirD_Originals/Train/"
path_predictions = "dir04_Predictions/{0}/".format(modelmode)
path_csv_loss = "dir05_CSV/{0}/Mask_Loss_{1}_{2}_{3}_{4}_{5}_{6}.csv".format(modelmode, modelmode, year, month, day,
                                                                             hour, minute)
path_csv_valloss = "dir05_CSV/{0}/Mask_Valloss_{1}_{2}_{3}_{4}_{5}_{6}.csv".format(modelmode, modelmode, year, month,
                                                                                   day, hour, minute)

# How many images will be predicted
number_of_predictions = 201

# How many images will be shown directly (They are saved anyway)
show_predictions = 0
