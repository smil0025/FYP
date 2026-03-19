The files in this image classifier have the following responsibilities:

input file (raw data):
- raw image data in folders to define class (material type)
- supplied to the prelim_img_processing.py function

prelim_img_processing:
- prepares input data to be suited to going into the classifier
    - converts to greyscale
    - resizes to 256x256 square with padding where required
    - augments the images (rotates 90, 180, 270 and flips horizontal and vertical)
        (this increases the amount of test data we have)
    - generates a second channel of data showing sobel edges (to be improved)
    - output of this function is the output and processed data folders (output is a duplicate, needs to be removed)
        (these images still have their class labels)

processed (dataset output by img processing):
- randomly splits the data it receives into
    Training: 70% (data to go to image classifier)
    Validation: 15% (images that will be used by classifier to see if it is working)
    Test: 15% (test data for us programmers to see how it goes on new data)
- This is supplied to the dataset.py file

dataset.py:
- reconfigures the processed data files to suit input to a pytorch image classifier
- supplies model.py

model.py:
- contains/defines the actual CNN (convolution layers) function

train.py:
- applies the CNN defined in model.py to our training data
- does a forward pass, computes loss using CrossEntropy
- Runs backprop and an optimiser
- does 10 iterations (epochs), outputs the loss and accuracy/validation score each time and a graph
  of the overall performance at the end

evaluate.py:
- Calculates overall model accuracy
- Calculates model accuracy per class (is it better at identifying one class than the other?)
- Generates a confusion matrix
- Lets us visualise the images it's identifying to see if there's a particular attribute tripping the model up
