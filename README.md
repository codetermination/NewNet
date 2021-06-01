# U-Net-in-Keras
Implementation of the U-Net-Model in Keras and Tensorflow.
The model is trained on images augmented by 3 differenty fence types. 
There are three datasets: 

1.) Simply augmented images ("NoBlur")

2.) Small blur of the fence border ("SingleBlur")

3.) Strong blur of the fence border ("DoubleBlur")

All images are from https://pix-zip.herokuapp.com. Each of the datasets is split into test and train. 

# Directory structure

In the project directory: 

* RunNewNet.py: File to train or test a model (Command see below)
* settings.py: Hyperparameter for training

dir01_moduldir: 

* Newnet.py: Code for the model
* constants.py: Hyperparameter


dir02_Dataset: 

* Contains images of 4 different datasets (NoBlur, SingleBlur, DoubleBlur, RealImgs)

dir03_Models:

* Place to store trained models. A model is saved with the start date and time of the training. There are three trained models provided: "NoBlur_2021_3_28_21_30", "SingleBlur_2021_3_29_10_20" and "DoubleBlur_2021_3_29_8_41".

dir04_Predictions: 

* Place to store the predictions. 

dir05_CSV:

* Place to store loss and vallos of a training. 

dir06_Calculations: 

* CalculatePSNR.py: File to calculate peak-signal-to-noise-ratio (PSNR) for a predicted dataset and an original dataset
* CalculateSSIM.py: File to calculate structural similarty (SSIM) for a predicted dataset and an original dataset 


# Test the model
There are three pre-trained models inside this directory. Per default the test of the "NoBlurModel" is executed. You can execute the test by: 

```python
python3 RunNewNet.py 
```
Execution must take place in the project directory. In settings.py you can change, which dataset you want to use


# Train the model
To train a new model, execute the following code in the project directory:

```python
python3 RunNewNet.py -t True
```
