import cv2
import glob
import numpy as np
from skimage import metrics


def calculate_ssim(counter, path_originals, path_predictions, ssim_csv_file):
    img_number = 1200 + counter
    # load the original image
    path_original_img = path_originals + str(img_number) + ".jpg"
    original = cv2.imread(path_original_img)

    # load the prdiction image
    path_prediction_img = path_predictions + str(img_number) + ".jpg"
    prediction = cv2.imread(path_prediction_img)

    # resize original to 272, 272
    original_resized = cv2.resize(original, (272, 272))

    # convert original to grayscale
    original_grayscale = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)

    # convert prediction to grayscale
    prediction_grayscale = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)

    # compare both of them with peak signal to noise rato
    ssim = metrics.peak_signal_noise_ratio(original_grayscale, prediction_grayscale)

    # add it to a csv file
    ssim_1d_array = np.atleast_1d(np.array(ssim))
    with open(ssim_csv_file, "ab") as f:
        np.savetxt(f, ssim_1d_array)
    print("ssim of image" + str(img_number) + ": " + str(ssim_1d_array))
    return ssim_1d_array


# csv file to write ssim:
ssim_csv_file = "ssim_NoBlur_2021_3_28_21_30.csv"  # Name of file with all ssim values
avg_ssim_csv_file = "avg_ssim_NoBlur_2021_3_28_21_30.csv"  # Name of file with avg ssim value

# Create a list containing all prediction images
path_predictions = "/home/felizia/NewNet/NewNet_Abgabe/dir04_Predictions/NoBlur/NoBlur_2021_3_28_21_30/Prediction"
all_predictions = glob.glob(path_predictions + "*.jpg")
print("all_predictions: " + str(len(all_predictions)))

# Create a list containing all original images
path_originals = "/home/felizia/NewNet/NewNet_Abgabe/dir02_Dataset/dirD_Originals/Test/OrigImgsClass/Vieh_orig"  # "/home/felizia/NewNet/Newnet_Abgabe/dir06_Test/Originals/Vieh_orig"
all_originals = glob.glob(path_originals + "*.jpg")
print("all_originals: " + str(len(all_originals)))

# Set counter to 1
counter = 1

# Create an ssim list with all values
ssim_sum = 0
for img in all_predictions:
    ssim_1d_array = calculate_ssim(counter, path_originals, path_predictions, ssim_csv_file)
    ssim_sum += ssim_1d_array
    counter += 1

# Calculate and save ssim avg
ssim_avg = ssim_sum / len(all_predictions)
ssim_avg_1d = np.atleast_1d(ssim_avg)

with open(avg_ssim_csv_file, "ab") as f:
    np.savetxt(f, ssim_avg_1d)

print("ssim average: " + str(ssim_avg))


