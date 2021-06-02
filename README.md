# U-Net-in-Keras
Implementation of the U-Net-Model<sup>1</sup> in Keras and Tensorflow.
The model is trained on images augmented by 3 differenty fence types. 

## Datasets
There are three datasets: 

1.) Simply augmented images ("NoBlur")
<p align="center"> 
<img src="https://user-images.githubusercontent.com/77172165/120307943-d7d6d200-c2d3-11eb-83ce-9eedc62fb21d.jpg" height="150"> <img src="https://user-images.githubusercontent.com/77172165/120308531-7cf1aa80-c2d4-11eb-9fea-3294ac3fd574.jpg" height = "150"> <img src="https://user-images.githubusercontent.com/77172165/120308728-b3c7c080-c2d4-11eb-9e41-e1ccb566c3b4.jpg" height ="150">


2.) Small blur of the fence border ("SingleBlur")
<p align="center">  
<img src="https://user-images.githubusercontent.com/77172165/120317375-d52daa00-c2de-11eb-9786-afa8b4548c8c.jpg" height = "150"> <img src="https://user-images.githubusercontent.com/77172165/120317533-05754880-c2df-11eb-89c8-91ffc04ceefc.jpg" height= "150"> <img src="https://user-images.githubusercontent.com/77172165/120317598-1faf2680-c2df-11eb-8aa2-703124ae17dd.jpg" height= "150">


3.) Strong blur of the fence border ("DoubleBlur")
<p align="center">  
<img src="https://user-images.githubusercontent.com/77172165/120317749-508f5b80-c2df-11eb-819b-e42300a6c01c.jpg" height="150"> <img src="https://user-images.githubusercontent.com/77172165/120317805-61d86800-c2df-11eb-94b2-ef17b712ea88.jpg" height="150"> <img src="https://user-images.githubusercontent.com/77172165/120317855-6ef55700-c2df-11eb-85e0-e4b0e15c4754.jpg" height="150">


All images are taken from https://pix-zip.herokuapp.com. Each of the datasets is split into test and train. 

## Directory structure

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


## Predictions
There are three pre-trained models inside this directory. Per default the test of the "NoBlurModel" is executed. You can execute the test by: 

```python
python3 RunNewNet.py 
```
Execution must take place in the project directory. In settings.py you can change the used dataset and model.

Here are three originals (left) and predictions (right) from the "NoBlur"-test-dataset:
<p align="center">
<img src="https://user-images.githubusercontent.com/77172165/120318162-d27f8480-c2df-11eb-91c2-8532e5abfd4c.jpg" height="150"> -> <img src="https://user-images.githubusercontent.com/77172165/120318371-107ca880-c2e0-11eb-84d5-a88d27da66b4.jpg" height="150">

<p align="center">
<img src= "https://user-images.githubusercontent.com/77172165/120318214-e1663700-c2df-11eb-9f73-48c590857aa1.jpg" height="150"> -> <img src= "https://user-images.githubusercontent.com/77172165/120318416-1d010100-c2e0-11eb-851c-abe39d86ba34.jpg" height="150">

<p align="center">
<img src="https://user-images.githubusercontent.com/77172165/120318274-f642ca80-c2df-11eb-8080-863f0ecf9da2.jpg" height="150"> -> <img src="https://user-images.githubusercontent.com/77172165/120318435-25593c00-c2e0-11eb-8811-910997e8d0ca.jpg" height = "150">


                                                                                                                 

 
## Train the model
To train a new model, execute the following code in the project directory:

```python
python3 RunNewNet.py -t True
```
Your model will be saved with the current date and time. The dataset can be changed in settings.py.
  
[1] Ronneberger, O. et al: <a href="https://arxiv.org/abs/1505.04597">U-Net: Convolutional Networks for Biomedical Image Segmentation</a>, 2015
