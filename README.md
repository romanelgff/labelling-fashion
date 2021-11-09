# CNN with Python (in progress)

Creating a Conventional Neuronal Networks that classify and label fashion images. We want networks that recognise the clothes, from **16,000 coloured images** (300x400 converted to 90x120) split into **3 main categories** (glasses, trousers, shoes) and **17 subclasses**. The objective is to return all labels associated with a given image. To solve the problem, we divided the task into smaller bits that we can solve independently (with different networks).

Neural networks don't work with image files, but with tensors. Therefore, we need to convert all the images into numpy arrays that Tensorflow can work with. We also need to resize the images, to save computational power and memory. To do so, we used the **image** module from the pillow library. The notebook file that does this job is called *9_3_convert-images-into-tensors.ipynb*.

This practical was done in the frame of a [365DataScience](https://learn.365datascience.com) course.

## Primary classification

We explored the labels and sublabels structure of our data, and decided to break it into several, smaller objectives that are solvable by a single model. The first of these is the primary classification, that is to label the image to one of three options (glasses/sunglasses, trousers/jeans, shoes). This was done in the *9_5_primary-classification-task-model.ipynb* file, in the *Primary_classification* folder.

To achieve this, we set up a relatively simple model consisting of: 2 convotional layers, 2 maxpool layers and the compulsory dense outcome layer. We decided to use this configuration because it keeps the training time lower, and allows us to check for different hyperparameters.

In terms of code, we imported datasets and preprocessed them, created the functions for hyperparameters tuning and confusion matrix and log the training process using TensorBoard.

We tried out the model with a filter size of 3, 5 and 7, and number of filters set to 32, 64, 96 and 128.

The results were impressive, with more than 99.9% accuracy accross all different combinations of hyerparamaters.

## Distinguishing glasses from sunglasses in glasses/sunglasses category

In the *9_5_primary-classification-task-model-1-glasses-sunglasses.ipynb* notebook (*Glasses_sunglasses* folder), we focus on the glasses/sunglasses category. We do exactly the same thing as for the primary classification, but instead we try to differentiate between 2 subcategories: glasses and sunglasses.

## Finding the best method for the trousers/jeans category

### Two different methods

Avalaible in *Separate* and *All* folders.

### Comparison of the methods

The comparison was done in the *9_11_comparing-trousers-jeans-techniques.ipynb* file. We attributed scores for both approaches. We test the models with the test set. If the model's predicted label is the good one then the model scores 1 point. 

The scores obtained for both models are very clore to each other (861 for the "all approach" against 860 for the "separate approach"), pratically the same, meaning that there is not significant benefit to one approach. In that case, the preferred method might be the one where the computational stress on the system and easier use it. That would be the network with four classes into one combined label.
