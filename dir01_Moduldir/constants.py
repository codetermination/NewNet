# Change the number of trainings images if you have more / less than 1400 to calculate the steps_per_epoch
number_training_images = 1400
image_input_size = 286
image_output_size = 272
number_of_epochs = 100
trainings_batch_size = 1
steps_per_epoch = int(number_training_images / trainings_batch_size)
val_split = 0.2
val_steps = 15
seed = 100
lernrate = 0.001
prediction_batch_size = 1

# Starting number of filters in Newnet
depth = 32
