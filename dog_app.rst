
.. code:: ipython3

    # Artificial Intelligence Nanodegree
    
    ## Convolutional Neural Networks
    
    ## Project: Write an Algorithm for a Dog Identification App 
    
    ---
    
    In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 
    
    > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
        "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
    
    In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
    
    >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.
    
    The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.
    
    
    
    ---
    ### Why We're Here 
    
    In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 
    
    ![Sample Dog Output](images/sample_dog_output.png)
    
    In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!
    
    ### The Road Ahead
    
    We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.
    
    * [Step 0](#step0): Import Datasets
    * [Step 1](#step1): Detect Humans
    * [Step 2](#step2): Detect Dogs
    * [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
    * [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
    * [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
    * [Step 6](#step6): Write your Algorithm
    * [Step 7](#step7): Test Your Algorithm
    
    ---
    <a id='step0'></a>
    ## Step 0: Import Datasets
    
    ### Import Dog Dataset
    
    In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
    - `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
    - `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
    - `dog_names` - list of string-valued dog breed names for translating labels

.. code:: ipython3

    from sklearn.datasets import load_files       
    from keras.utils import np_utils
    import numpy as np
    from glob import glob
    
    # define function to load train, test, and validation datasets
    def load_dataset(path):
        data = load_files(path)
        dog_files = np.array(data['filenames'])
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets
    
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('dogImages/train')
    valid_files, valid_targets = load_dataset('dogImages/valid')
    test_files, test_targets = load_dataset('dogImages/test')
    
    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
    
    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))


.. parsed-literal::

    /anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


.. parsed-literal::

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


Import Human Dataset
~~~~~~~~~~~~~~~~~~~~

In the code cell below, we import a dataset of human images, where the
file paths are stored in the numpy array ``human_files``.

.. code:: ipython3

    import random
    random.seed(8675309)
    
    # load filenames in shuffled human dataset
    human_files = np.array(glob("lfw/*/*"))
    random.shuffle(human_files)
    
    # print statistics about the dataset
    print('There are %d total human images.' % len(human_files))


.. parsed-literal::

    There are 13233 total human images.


--------------

 ## Step 1: Detect Humans

We use OpenCV's implementation of `Haar feature-based cascade
classifiers <http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html>`__
to detect human faces in images. OpenCV provides many pre-trained face
detectors, stored as XML files on
`github <https://github.com/opencv/opencv/tree/master/data/haarcascades>`__.
We have downloaded one of these detectors and stored it in the
``haarcascades`` directory.

In the next code cell, we demonstrate how to use this detector to find
human faces in a sample image.

.. code:: ipython3

    import cv2                
    import matplotlib.pyplot as plt                        
    %matplotlib inline                               
    
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    
    # load color (BGR) image
    img = cv2.imread(human_files[3])
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # find faces in image
    faces = face_cascade.detectMultiScale(gray)
    
    # print number of faces detected in the image
    print('Number of faces detected:', len(faces))
    
    # get bounding box for each detected face
    for (x,y,w,h) in faces:
        # add bounding box to color image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # display the image, along with bounding box
    plt.imshow(cv_rgb)
    plt.show()


.. parsed-literal::

    Number of faces detected: 1



.. image:: output_5_1.png


Before using any of the face detectors, it is standard procedure to
convert the images to grayscale. The ``detectMultiScale`` function
executes the classifier stored in ``face_cascade`` and takes the
grayscale image as a parameter.

In the above code, ``faces`` is a numpy array of detected faces, where
each row corresponds to a detected face. Each detected face is a 1D
array with four entries that specifies the bounding box of the detected
face. The first two entries in the array (extracted in the above code as
``x`` and ``y``) specify the horizontal and vertical positions of the
top left corner of the bounding box. The last two entries in the array
(extracted here as ``w`` and ``h``) specify the width and height of the
box.

Write a Human Face Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use this procedure to write a function that returns ``True`` if a
human face is detected in an image and ``False`` otherwise. This
function, aptly named ``face_detector``, takes a string-valued file path
to an image as input and appears in the code block below.

.. code:: ipython3

    # returns "True" if face is detected in image stored at img_path
    def face_detector(human_files):
            img = cv2.imread(human_files)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)
            return len(faces) > 0

(IMPLEMENTATION) Assess the Human Face Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| **Question 1:** Use the code cell below to test the performance of the
  ``face_detector`` function.
| - What percentage of the first 100 images in ``human_files`` have a
  detected human face?
| - What percentage of the first 100 images in ``dog_files`` have a
  detected human face?

Ideally, we would like 100% of human images with a detected face and 0%
of dog images with a detected face. You will see that our algorithm
falls short of this goal, but still gives acceptable performance. We
extract the file paths for the first 100 images from each of the
datasets and store them in the numpy arrays ``human_files_short`` and
``dog_files_short``.

**Answer:**

.. code:: ipython3

    human_files_short = human_files[:100]
    dog_files_short = train_files[:100]
    
    humans_faces_detected=0
    for img_path in (human_files_short):
        humans_faces_detected+=face_detector(img_path)
    dogs_detected=0
    for img_path in (dog_files_short):
        dogs_detected+=face_detector(img_path)
    print("Human Percentage: {}".format(np.round(humans_faces_detected/100, 2)))
    print("Dog Percentage: {}".format(np.round(dogs_detected/100, 2)))


.. parsed-literal::

    Human Percentage: 0.98
    Dog Percentage: 0.12


**Question 2:** This algorithmic choice necessitates that we communicate
to the user that we accept human images only when they provide a clear
view of a face (otherwise, we risk having unneccessarily frustrated
users!). In your opinion, is this a reasonable expectation to pose on
the user? If not, can you think of a way to detect humans in images that
does not necessitate an image with a clearly presented face?

**Answer:**

We suggest the face detector from OpenCV as a potential way to detect
human images in your algorithm, but you are free to explore other
approaches, especially approaches that make use of deep learning :).
Please use the code cell below to design and test your own face
detection algorithm. If you decide to pursue this *optional* task,
report performance on each of the datasets.

.. code:: ipython3

    ## (Optional) TODO: Report the performance of another  
    ## face detection algorithm on the LFW dataset
    ### Feel free to use as many code cells as needed.
    #import VGG16 without last fully connected layer
    #import weights from imagenet  


--------------

 ## Step 2: Detect Dogs

In this section, we use a pre-trained
`ResNet-50 <http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006>`__
model to detect dogs in images. Our first line of code downloads the
ResNet-50 model, along with weights that have been trained on
`ImageNet <http://www.image-net.org/>`__, a very large, very popular
dataset used for image classification and other vision tasks. ImageNet
contains over 10 million URLs, each linking to an image containing an
object from one of `1000
categories <https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a>`__.
Given an image, this pre-trained ResNet-50 model returns a prediction
(derived from the available categories in ImageNet) for the object that
is contained in the image.

.. code:: ipython3

    from keras.applications.resnet50 import ResNet50
    
    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')

Pre-process the Data
~~~~~~~~~~~~~~~~~~~~

When using TensorFlow as backend, Keras CNNs require a 4D array (which
we'll also refer to as a 4D tensor) as input, with shape

.. math::


   (\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),

where ``nb_samples`` corresponds to the total number of images (or
samples), and ``rows``, ``columns``, and ``channels`` correspond to the
number of rows, columns, and channels for each image, respectively.

The ``path_to_tensor`` function below takes a string-valued file path to
a color image as input and returns a 4D tensor suitable for supplying to
a Keras CNN. The function first loads the image and resizes it to a
square image that is :math:`224 \times 224` pixels. Next, the image is
converted to an array, which is then resized to a 4D tensor. In this
case, since we are working with color images, each image has three
channels. Likewise, since we are processing a single image (or sample),
the returned tensor will always have shape

.. math::


   (1, 224, 224, 3).

The ``paths_to_tensor`` function takes a numpy array of string-valued
image paths as input and returns a 4D tensor with shape

.. math::


   (\text{nb_samples}, 224, 224, 3).

Here, ``nb_samples`` is the number of samples, or number of images, in
the supplied array of image paths. It is best to think of ``nb_samples``
as the number of 3D tensors (where each 3D tensor corresponds to a
different image) in your dataset!

.. code:: ipython3

    from keras.preprocessing import image                  
    from tqdm import tqdm
    
    def path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)
    
    def paths_to_tensor(img_paths):
        list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

Making Predictions with ResNet-50
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained
model in Keras, requires some additional processing. First, the RGB
image is converted to BGR by reordering the channels. All pre-trained
models have the additional normalization step that the mean pixel
(expressed in RGB as :math:`[103.939, 116.779, 123.68]` and calculated
from all pixels in all images in ImageNet) must be subtracted from every
pixel in each image. This is implemented in the imported function
``preprocess_input``. If you're curious, you can check the code for
``preprocess_input``
`here <https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py>`__.

Now that we have a way to format our image for supplying to ResNet-50,
we are now ready to use the model to extract the predictions. This is
accomplished with the ``predict`` method, which returns an array whose
:math:`i`-th entry is the model's predicted probability that the image
belongs to the :math:`i`-th ImageNet category. This is implemented in
the ``ResNet50_predict_labels`` function below.

By taking the argmax of the predicted probability vector, we obtain an
integer corresponding to the model's predicted object class, which we
can identify with an object category through the use of this
`dictionary <https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a>`__.

.. code:: ipython3

    from keras.applications.resnet50 import preprocess_input, decode_predictions
    
    def ResNet50_predict_labels(img_path):
        # returns prediction vector for image located at img_path
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(ResNet50_model.predict(img))

Write a Dog Detector
~~~~~~~~~~~~~~~~~~~~

While looking at the
`dictionary <https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a>`__,
you will notice that the categories corresponding to dogs appear in an
uninterrupted sequence and correspond to dictionary keys 151-268,
inclusive, to include all categories from ``'Chihuahua'`` to
``'Mexican hairless'``. Thus, in order to check to see if an image is
predicted to contain a dog by the pre-trained ResNet-50 model, we need
only check if the ``ResNet50_predict_labels`` function above returns a
value between 151 and 268 (inclusive).

We use these ideas to complete the ``dog_detector`` function below,
which returns ``True`` if a dog is detected in an image (and ``False``
if not).

.. code:: ipython3

    ### returns "True" if a dog is detected in the image stored at img_path
    def dog_detector(img_path):
        prediction = ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151)) 

(IMPLEMENTATION) Assess the Dog Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| **Question 3:** Use the code cell below to test the performance of
  your ``dog_detector`` function.
| - What percentage of the images in ``human_files_short`` have a
  detected dog?
| - What percentage of the images in ``dog_files_short`` have a detected
  dog?

**Answer:**

.. code:: ipython3

    ### TODO: Test the performance of the dog_detector function
    ### on the images in human_files_short and dog_files_short.
    
    human_files_short = human_files[:100]
    dog_files_short = train_files[:100]
    
    humans_faces_detected=0
    for img_path in (human_files_short):
        humans_faces_detected+=dog_detector(img_path)
    dogs_detected=0
    for img_path in (dog_files_short):
        dogs_detected+=dog_detector(img_path)
    print("Human Percentage: {}".format(np.round(humans_faces_detected/100, 2)))
    print("Dog Percentage: {}".format(np.round(dogs_detected/100, 2)))
    



.. parsed-literal::

    Human Percentage: 0.01
    Dog Percentage: 1.0


--------------

 ## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we
need a way to predict breed from images. In this step, you will create a
CNN that classifies dog breeds. You must create your CNN *from scratch*
(so, you can't use transfer learning *yet*!), and you must attain a test
accuracy of at least 1%. In Step 5 of this notebook, you will have the
opportunity to use transfer learning to create a CNN that attains
greatly improved accuracy.

Be careful with adding too many trainable layers! More parameters means
longer training, which means you are more likely to need a GPU to
accelerate the training process. Thankfully, Keras provides a handy
estimate of the time that each epoch is likely to take; you can
extrapolate this estimate to figure out how long it will take for your
algorithm to train.

We mention that the task of assigning breed to dogs from images is
considered exceptionally challenging. To see why, consider that *even a
human* would have great difficulty in distinguishing between a Brittany
and a Welsh Springer Spaniel.

+------------+--------------------------+
| Brittany   | Welsh Springer Spaniel   |
+============+==========================+
+------------+--------------------------+

It is not difficult to find other dog breed pairs with minimal
inter-class variation (for instance, Curly-Coated Retrievers and
American Water Spaniels).

+--------------------------+--------------------------+
| Curly-Coated Retriever   | American Water Spaniel   |
+==========================+==========================+
+--------------------------+--------------------------+

Likewise, recall that labradors come in yellow, chocolate, and black.
Your vision-based algorithm will have to conquer this high intra-class
variation to determine how to classify all of these different shades as
the same breed.

+-------------------+----------------------+
| Yellow Labrador   | Chocolate Labrador   |
+===================+======================+
+-------------------+----------------------+

We also mention that random chance presents an exceptionally low bar:
setting aside the fact that the classes are slightly imabalanced, a
random guess will provide a correct answer roughly 1 in 133 times, which
corresponds to an accuracy of less than 1%.

Remember that the practice is far ahead of the theory in deep learning.
Experiment with many different architectures, and trust your intuition.
And, of course, have fun!

Pre-process the Data
~~~~~~~~~~~~~~~~~~~~

We rescale the images by dividing every pixel in every image by 255.

.. code:: ipython3

    from keras.preprocessing import image                  
    from tqdm import tqdm
    
    def path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)
    
    def paths_to_tensor(img_paths):
        list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

.. code:: ipython3

    from PIL import ImageFile                            
    ImageFile.LOAD_TRUNCATED_IMAGES = True                 
    
    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32')/255
    valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
    test_tensors = paths_to_tensor(test_files).astype('float32')/255


.. parsed-literal::

    100%|██████████| 6680/6680 [01:58<00:00, 56.44it/s]
    100%|██████████| 835/835 [00:13<00:00, 61.70it/s]
    100%|██████████| 836/836 [00:12<00:00, 64.41it/s]


.. code:: ipython3

    from keras.preprocessing import image                  
    from tqdm import tqdm
    
    def path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)
    
    def paths_to_tensor(img_paths):
        list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

(IMPLEMENTATION) Model Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a CNN to classify dog breed. At the end of your code cell block,
summarize the layers of your model by executing the line:

::

        model.summary()

We have imported some Python modules to get you started, but feel free
to import as many modules as you need. If you end up getting stuck,
here's a hint that specifies a model that trains relatively fast on CPU
and attains >1% test accuracy in 5 epochs:

.. figure:: images/sample_cnn.png
   :alt: Sample CNN

   Sample CNN

**Question 4:** Outline the steps you took to get to your final CNN
architecture and your reasoning at each step. If you chose to use the
hinted architecture above, describe why you think that CNN architecture
should work well for the image classification task.

**Answer:**

.. code:: ipython3

    from sklearn.datasets import load_files       
    from keras.utils import np_utils
    import numpy as np
    from glob import glob
    
    # define function to load train, test, and validation datasets
    def load_dataset(path):
        data = load_files(path)
        dog_files = np.array(data['filenames'])
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets
    
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('dogImages/train')
    valid_files, valid_targets = load_dataset('dogImages/valid')
    test_files, test_targets = load_dataset('dogImages/test')
    
    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
    
    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))


.. parsed-literal::

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


.. code:: ipython3

    from keras import layers
    from keras import models
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.layers import Dropout, Flatten, Dense
    from keras.models import Sequential
    
    model = Sequential()
    
    ### TODO: Define your architecture.
    model.add(layers.Conv2D(16, (2, 2), activation='relu',
                            input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #GlobalAveragePooling to minimize overfitting, h*w*d is reduced to 1*1*d; each h*w feature map 
    #is reduced to a single number by taking the average of all h*w values
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
              
    model.add(layers.Dense(133, activation='relu'))
    
    
    model.summary()


.. parsed-literal::

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 223, 223, 16)      208       
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 111, 111, 16)      0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 110, 110, 32)      2080      
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 55, 55, 32)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 54, 54, 64)        8256      
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 27, 27, 64)        0         
    _________________________________________________________________
    global_average_pooling2d_2 ( (None, 64)                0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               8645      
    =================================================================
    Total params: 19,189
    Trainable params: 19,189
    Non-trainable params: 0
    _________________________________________________________________


Compile the Model
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

(IMPLEMENTATION) Train the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train your model in the code cell below. Use model checkpointing to save
the model that attains the best validation loss.

You are welcome to `augment the training
data <https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>`__,
but this is not a requirement.

.. code:: ipython3

    from keras.callbacks import ModelCheckpoint  
    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator
    from PIL import ImageFile                            
    ImageFile.LOAD_TRUNCATED_IMAGES = True                 
    
    ### TODO: specify the number of epochs that you would like to use to train the model.
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_files,
        target_size=(228, 228),
        batch_size=32,
        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
        valid_files,
        target_size=(228, 228),
        batch_size=32,
        class_mode='categorical')
    
    history = model.fit_generator(
          train_generator,
          steps_per_epoch=100,
          epochs=20,
          validation_data=validation_generator,
          validation_steps=50)
    
    
    
    ### Do NOT modify the code below this line.
    
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                                   verbose=1, save_best_only=True)
    
    model.fit(train_tensors, train_targets, 
              validation_data=(valid_tensors, valid_targets),
              epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


::


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-30-72ca83cd1114> in <module>()
         22     target_size=(228, 228),
         23     batch_size=32,
    ---> 24     class_mode='categorical')
         25 
         26 validation_generator = test_datagen.flow_from_directory(


    /anaconda3/lib/python3.6/site-packages/keras/preprocessing/image.py in flow_from_directory(self, directory, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation)
        570             follow_links=follow_links,
        571             subset=subset,
    --> 572             interpolation=interpolation)
        573 
        574     def standardize(self, x):


    /anaconda3/lib/python3.6/site-packages/keras/preprocessing/image.py in __init__(self, directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation)
       1183         if not classes:
       1184             classes = []
    -> 1185             for subdir in sorted(os.listdir(directory)):
       1186                 if os.path.isdir(os.path.join(directory, subdir)):
       1187                     classes.append(subdir)


    ValueError: listdir: embedded null character in path


Load the Model with the Best Validation Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model.load_weights('saved_models/weights.best.from_scratch.hdf5')


::


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-32-b1224892bd7d> in <module>()
    ----> 1 model.load_weights('saved_models/weights.best.from_scratch.hdf5')
    

    /anaconda3/lib/python3.6/site-packages/keras/models.py in load_weights(self, filepath, by_name, skip_mismatch, reshape)
        722         if h5py is None:
        723             raise ImportError('`load_weights` requires h5py.')
    --> 724         with h5py.File(filepath, mode='r') as f:
        725             if 'layer_names' not in f.attrs and 'model_weights' in f:
        726                 f = f['model_weights']


    /anaconda3/lib/python3.6/site-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, **kwds)
        267             with phil:
        268                 fapl = make_fapl(driver, libver, **kwds)
    --> 269                 fid = make_fid(name, mode, userblock_size, fapl, swmr=swmr)
        270 
        271                 if swmr_support:


    /anaconda3/lib/python3.6/site-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
         97         if swmr and swmr_support:
         98             flags |= h5f.ACC_SWMR_READ
    ---> 99         fid = h5f.open(name, flags, fapl=fapl)
        100     elif mode == 'r+':
        101         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)


    h5py/_objects.pyx in h5py._objects.with_phil.wrapper()


    h5py/_objects.pyx in h5py._objects.with_phil.wrapper()


    h5py/h5f.pyx in h5py.h5f.open()


    OSError: Unable to open file (unable to open file: name = 'saved_models/weights.best.from_scratch.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)


Test the Model
~~~~~~~~~~~~~~

Try out your model on the test dataset of dog images. Ensure that your
test accuracy is greater than 1%.

.. code:: ipython3

    # get index of predicted dog breed for each image in test set
    dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    
    # report test accuracy
    test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


::


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-8c1b03961dad> in <module>()
          1 # get index of predicted dog breed for each image in test set
    ----> 2 dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
          3 
          4 # report test accuracy
          5 test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)


    NameError: name 'test_tensors' is not defined


--------------

 ## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to
train a CNN using transfer learning. In the following step, you will get
a chance to use transfer learning to train your own CNN.

Obtain Bottleneck Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import numpy as np
    bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
    train_VGG16 = bottleneck_features['train']
    valid_VGG16 = bottleneck_features['valid']
    test_VGG16 = bottleneck_features['test']

Model Architecture
~~~~~~~~~~~~~~~~~~

The model uses the the pre-trained VGG-16 model as a fixed feature
extractor, where the last convolutional output of VGG-16 is fed as input
to our model. We only add a global average pooling layer and a fully
connected layer, where the latter contains one node for each dog
category and is equipped with a softmax.

.. code:: ipython3

    from keras import layers
    from keras import models
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.layers import Dropout, Dense
    from keras.models import Sequential
    from keras.applications import VGG16
    
    VGG16_model= Sequential()
    VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
    VGG16_model.add(Dense(133, activation='softmax'))
    
    VGG16_model.summary()


.. parsed-literal::

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_4 ( (None, 512)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________


Compile the Model
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from keras import optimizers
    
    rmsprop = optimizers.RMSprop(lr=5*10^-6)
    VGG16_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    #VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop',' metrics=['accuracy'])


Train the Model
~~~~~~~~~~~~~~~

.. code:: ipython3

    from sklearn.datasets import load_files       
    from keras.utils import np_utils
    import numpy as np
    from glob import glob
    
    # define function to load train, test, and validation datasets
    def load_dataset(path):
        data = load_files(path)
        dog_files = np.array(data['filenames'])
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets
    
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('dogImages/train')
    valid_files, valid_targets = load_dataset('dogImages/valid')
    test_files, test_targets = load_dataset('dogImages/test')
    
    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
    
    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))


.. parsed-literal::

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


.. code:: ipython3

    ### TODO: Train the model.
    from keras.callbacks import ModelCheckpoint
    
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                                   verbose=1, save_best_only=True)
    
    history=VGG16_model.fit(train_VGG16, train_targets, 
              validation_data=(valid_VGG16, valid_targets),
              epochs=50, batch_size=20, callbacks=[checkpointer], verbose=1)


.. parsed-literal::

    Train on 6680 samples, validate on 835 samples
    Epoch 1/50
    6680/6680 [==============================] - 4s 583us/step - loss: 4.3712 - acc: 0.7287 - val_loss: 5.8742 - val_acc: 0.5569
    
    Epoch 00001: val_loss improved from inf to 5.87424, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 2/50
    6680/6680 [==============================] - 2s 256us/step - loss: 4.3695 - acc: 0.7284 - val_loss: 5.8824 - val_acc: 0.5569
    
    Epoch 00002: val_loss did not improve
    Epoch 3/50
    6680/6680 [==============================] - 2s 256us/step - loss: 4.3707 - acc: 0.7284 - val_loss: 5.8794 - val_acc: 0.5605
    
    Epoch 00003: val_loss did not improve
    Epoch 4/50
    6680/6680 [==============================] - 2s 257us/step - loss: 4.3698 - acc: 0.7286 - val_loss: 5.8785 - val_acc: 0.5545
    
    Epoch 00004: val_loss did not improve
    Epoch 5/50
    6680/6680 [==============================] - 2s 250us/step - loss: 4.3682 - acc: 0.7286 - val_loss: 5.8851 - val_acc: 0.5581
    
    Epoch 00005: val_loss did not improve
    Epoch 6/50
    6680/6680 [==============================] - 2s 259us/step - loss: 4.3707 - acc: 0.7286 - val_loss: 5.8884 - val_acc: 0.5593
    
    Epoch 00006: val_loss did not improve
    Epoch 7/50
    6680/6680 [==============================] - 2s 262us/step - loss: 4.3712 - acc: 0.7287 - val_loss: 5.8791 - val_acc: 0.5605
    
    Epoch 00007: val_loss did not improve
    Epoch 8/50
    6680/6680 [==============================] - 2s 269us/step - loss: 4.3683 - acc: 0.7283 - val_loss: 5.8810 - val_acc: 0.5569
    
    Epoch 00008: val_loss did not improve
    Epoch 9/50
    6680/6680 [==============================] - 2s 253us/step - loss: 4.3699 - acc: 0.7287 - val_loss: 5.8837 - val_acc: 0.5581
    
    Epoch 00009: val_loss did not improve
    Epoch 10/50
    6680/6680 [==============================] - 2s 284us/step - loss: 4.3695 - acc: 0.7287 - val_loss: 5.8878 - val_acc: 0.5629
    
    Epoch 00010: val_loss did not improve
    Epoch 11/50
    6680/6680 [==============================] - 2s 248us/step - loss: 4.3696 - acc: 0.7286 - val_loss: 5.8967 - val_acc: 0.5593
    
    Epoch 00011: val_loss did not improve
    Epoch 12/50
    6680/6680 [==============================] - 2s 281us/step - loss: 4.3687 - acc: 0.7286 - val_loss: 5.8931 - val_acc: 0.5569
    
    Epoch 00012: val_loss did not improve
    Epoch 13/50
    6680/6680 [==============================] - 2s 266us/step - loss: 4.3695 - acc: 0.7286 - val_loss: 5.8999 - val_acc: 0.5557
    
    Epoch 00013: val_loss did not improve
    Epoch 14/50
    6680/6680 [==============================] - 2s 275us/step - loss: 4.3684 - acc: 0.7289 - val_loss: 5.9028 - val_acc: 0.5581
    
    Epoch 00014: val_loss did not improve
    Epoch 15/50
    6680/6680 [==============================] - 2s 260us/step - loss: 4.3694 - acc: 0.7287 - val_loss: 5.8870 - val_acc: 0.5581
    
    Epoch 00015: val_loss did not improve
    Epoch 16/50
    6680/6680 [==============================] - 2s 269us/step - loss: 4.3678 - acc: 0.7287 - val_loss: 5.8997 - val_acc: 0.5605
    
    Epoch 00016: val_loss did not improve
    Epoch 17/50
    6680/6680 [==============================] - 2s 262us/step - loss: 4.3709 - acc: 0.7287 - val_loss: 5.8916 - val_acc: 0.5593
    
    Epoch 00017: val_loss did not improve
    Epoch 18/50
    6680/6680 [==============================] - 2s 267us/step - loss: 4.3708 - acc: 0.7286 - val_loss: 5.8961 - val_acc: 0.5557
    
    Epoch 00018: val_loss did not improve
    Epoch 19/50
    6680/6680 [==============================] - 2s 260us/step - loss: 4.3678 - acc: 0.7287 - val_loss: 5.9215 - val_acc: 0.5521
    
    Epoch 00019: val_loss did not improve
    Epoch 20/50
    6680/6680 [==============================] - 2s 264us/step - loss: 4.3694 - acc: 0.7286 - val_loss: 5.9078 - val_acc: 0.5545
    
    Epoch 00020: val_loss did not improve
    Epoch 21/50
    6680/6680 [==============================] - 2s 265us/step - loss: 4.3682 - acc: 0.7287 - val_loss: 5.9136 - val_acc: 0.5569
    
    Epoch 00021: val_loss did not improve
    Epoch 22/50
    6680/6680 [==============================] - 2s 261us/step - loss: 4.3687 - acc: 0.7287 - val_loss: 5.8981 - val_acc: 0.5569
    
    Epoch 00022: val_loss did not improve
    Epoch 23/50
    6680/6680 [==============================] - 2s 273us/step - loss: 4.3717 - acc: 0.7286 - val_loss: 5.8985 - val_acc: 0.5593
    
    Epoch 00023: val_loss did not improve
    Epoch 24/50
    6680/6680 [==============================] - 2s 263us/step - loss: 4.3690 - acc: 0.7283 - val_loss: 5.9022 - val_acc: 0.5581
    
    Epoch 00024: val_loss did not improve
    Epoch 25/50
    6680/6680 [==============================] - 2s 266us/step - loss: 4.3688 - acc: 0.7284 - val_loss: 5.9026 - val_acc: 0.5545
    
    Epoch 00025: val_loss did not improve
    Epoch 26/50
    6680/6680 [==============================] - 2s 259us/step - loss: 4.3711 - acc: 0.7286 - val_loss: 5.8846 - val_acc: 0.5629
    
    Epoch 00026: val_loss did not improve
    Epoch 27/50
    6680/6680 [==============================] - 2s 259us/step - loss: 4.3696 - acc: 0.7286 - val_loss: 5.8935 - val_acc: 0.5569
    
    Epoch 00027: val_loss did not improve
    Epoch 28/50
    6680/6680 [==============================] - 2s 254us/step - loss: 4.3688 - acc: 0.7286 - val_loss: 5.9175 - val_acc: 0.5545
    
    Epoch 00028: val_loss did not improve
    Epoch 29/50
    6680/6680 [==============================] - 2s 250us/step - loss: 4.3687 - acc: 0.7286 - val_loss: 5.9062 - val_acc: 0.5533
    
    Epoch 00029: val_loss did not improve
    Epoch 30/50
    6680/6680 [==============================] - 2s 248us/step - loss: 4.3696 - acc: 0.7286 - val_loss: 5.9150 - val_acc: 0.5569
    
    Epoch 00030: val_loss did not improve
    Epoch 31/50
    6680/6680 [==============================] - 2s 249us/step - loss: 4.3703 - acc: 0.7286 - val_loss: 5.9116 - val_acc: 0.5545
    
    Epoch 00031: val_loss did not improve
    Epoch 32/50
    6680/6680 [==============================] - 2s 263us/step - loss: 4.3704 - acc: 0.7286 - val_loss: 5.9066 - val_acc: 0.5521
    
    Epoch 00032: val_loss did not improve
    Epoch 33/50
    6680/6680 [==============================] - 2s 247us/step - loss: 4.3706 - acc: 0.7287 - val_loss: 5.9118 - val_acc: 0.5545
    
    Epoch 00033: val_loss did not improve
    Epoch 34/50
    6680/6680 [==============================] - 2s 251us/step - loss: 4.3698 - acc: 0.7284 - val_loss: 5.9223 - val_acc: 0.5545
    
    Epoch 00034: val_loss did not improve
    Epoch 35/50
    6680/6680 [==============================] - 2s 258us/step - loss: 4.3683 - acc: 0.7287 - val_loss: 5.9289 - val_acc: 0.5509
    
    Epoch 00035: val_loss did not improve
    Epoch 36/50
    6680/6680 [==============================] - 2s 266us/step - loss: 4.3707 - acc: 0.7287 - val_loss: 5.9070 - val_acc: 0.5545
    
    Epoch 00036: val_loss did not improve
    Epoch 37/50
    6680/6680 [==============================] - 2s 276us/step - loss: 4.3694 - acc: 0.7287 - val_loss: 5.9169 - val_acc: 0.5533
    
    Epoch 00037: val_loss did not improve
    Epoch 38/50
    6680/6680 [==============================] - 2s 272us/step - loss: 4.3696 - acc: 0.7284 - val_loss: 5.9236 - val_acc: 0.5533
    
    Epoch 00038: val_loss did not improve
    Epoch 39/50
    6680/6680 [==============================] - 2s 254us/step - loss: 4.3679 - acc: 0.7287 - val_loss: 5.9126 - val_acc: 0.5569
    
    Epoch 00039: val_loss did not improve
    Epoch 40/50
    6680/6680 [==============================] - 2s 265us/step - loss: 4.3676 - acc: 0.7286 - val_loss: 5.9216 - val_acc: 0.5533
    
    Epoch 00040: val_loss did not improve
    Epoch 41/50
    6680/6680 [==============================] - 2s 268us/step - loss: 4.3694 - acc: 0.7286 - val_loss: 5.9039 - val_acc: 0.5569
    
    Epoch 00041: val_loss did not improve
    Epoch 42/50
    6680/6680 [==============================] - 2s 270us/step - loss: 4.3698 - acc: 0.7284 - val_loss: 5.9101 - val_acc: 0.5557
    
    Epoch 00042: val_loss did not improve
    Epoch 43/50
    6680/6680 [==============================] - 2s 262us/step - loss: 4.3686 - acc: 0.7286 - val_loss: 5.9226 - val_acc: 0.5545
    
    Epoch 00043: val_loss did not improve
    Epoch 44/50
    6680/6680 [==============================] - 2s 261us/step - loss: 4.3694 - acc: 0.7286 - val_loss: 5.9109 - val_acc: 0.5581
    
    Epoch 00044: val_loss did not improve
    Epoch 45/50
    6680/6680 [==============================] - 2s 249us/step - loss: 4.3679 - acc: 0.7287 - val_loss: 5.9244 - val_acc: 0.5521
    
    Epoch 00045: val_loss did not improve
    Epoch 46/50
    6680/6680 [==============================] - 2s 290us/step - loss: 4.3702 - acc: 0.7284 - val_loss: 5.9019 - val_acc: 0.5569
    
    Epoch 00046: val_loss did not improve
    Epoch 47/50
    6680/6680 [==============================] - 2s 322us/step - loss: 4.3684 - acc: 0.7286 - val_loss: 5.9055 - val_acc: 0.5557
    
    Epoch 00047: val_loss did not improve
    Epoch 48/50
    6680/6680 [==============================] - 2s 367us/step - loss: 4.3704 - acc: 0.7284 - val_loss: 5.9057 - val_acc: 0.5545
    
    Epoch 00048: val_loss did not improve
    Epoch 49/50
    6680/6680 [==============================] - 2s 349us/step - loss: 4.3689 - acc: 0.7284 - val_loss: 5.9053 - val_acc: 0.5569
    
    Epoch 00049: val_loss did not improve
    Epoch 50/50
    6680/6680 [==============================] - 2s 306us/step - loss: 4.3686 - acc: 0.7286 - val_loss: 5.9091 - val_acc: 0.5557
    
    Epoch 00050: val_loss did not improve


Load the Model with the Best Validation Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()



.. image:: output_47_0.png



.. image:: output_47_1.png


.. code:: ipython3

    VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')

Test the Model
~~~~~~~~~~~~~~

Now, we can use the CNN to test how well it identifies breed within our
test dataset of dog images. We print the test accuracy below.

.. code:: ipython3

    import numpy as np
    # get index of predicted dog breed for each image in test set
    VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]
    
    # report test accuracy
    test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)



.. parsed-literal::

    Test accuracy: 56.1005%


.. code:: ipython3

    from extract_bottleneck_features import *
    
    def VGG16_predict_breed(img_path):
        # extract bottleneck features
        bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = VGG16_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]
    
    


--------------

 ## Step 5: Create a CNN to Classify Dog Breeds (using Transfer
Learning)

You will now use transfer learning to create a CNN that can identify dog
breed from images. Your CNN must attain at least 60% accuracy on the
test set.

In Step 4, we used transfer learning to create a CNN using VGG-16
bottleneck features. In this section, you must use the bottleneck
features from a different pre-trained model. To make things easier for
you, we have pre-computed the features for all of the networks that are
currently available in Keras: -
`VGG-19 <https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz>`__
bottleneck features -
`ResNet-50 <https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz>`__
bottleneck features -
`Inception <https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz>`__
bottleneck features -
`Xception <https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz>`__
bottleneck features

The files are encoded as such:

::

    Dog{network}Data.npz

where ``{network}``, in the above filename, can be one of ``VGG19``,
``Resnet50``, ``InceptionV3``, or ``Xception``. Pick one of the above
architectures, download the corresponding bottleneck features, and store
the downloaded file in the ``bottleneck_features/`` folder in the
repository.

(IMPLEMENTATION) Obtain Bottleneck Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the code block below, extract the bottleneck features corresponding
to the train, test, and validation sets by running the following:

::

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']

Predict Dog Breed with the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import os
    import numpy as np
    
    ### TODO: Obtain bottleneck features from another pre-trained CNN.
    bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
    train_VGG19 = bottleneck_features['train']
    valid_VGG19 = bottleneck_features['valid']
    test_VGG19 = bottleneck_features['test']


.. code:: ipython3

    from sklearn.datasets import load_files       
    from keras.utils import np_utils
    import numpy as np
    from glob import glob
    
    # define function to load train, test, and validation datasets
    def load_dataset(path):
        data = load_files(path)
        dog_files = np.array(data['filenames'])
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets
    
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('dogImages/train')
    valid_files, valid_targets = load_dataset('dogImages/valid')
    test_files, test_targets = load_dataset('dogImages/test')
    
    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
    
    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))


.. parsed-literal::

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


(IMPLEMENTATION) Model Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a CNN to classify dog breed. At the end of your code cell block,
summarize the layers of your model by executing the line:

::

        <your model's name>.summary()

**Question 5:** Outline the steps you took to get to your final CNN
architecture and your reasoning at each step. Describe why you think the
architecture is suitable for the current problem.

**Answer:**

.. code:: ipython3

    ### TODO: Define your architecture.
    from keras import layers
    from keras import models
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.layers import Dropout, Flatten, Dense
    from keras.models import Sequential
    from keras.applications import VGG19
    
    VGG19_model = Sequential()
    VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
    
    VGG19_model.add(layers.Dense(133, activation='softmax'))
    
    
    VGG19_model.summary()


.. parsed-literal::

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_6 ( (None, 512)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________


(IMPLEMENTATION) Compile the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ### TODO: Compile the model.
    VGG19_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

(IMPLEMENTATION) Train the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train your model in the code cell below. Use model checkpointing to save
the model that attains the best validation loss.

You are welcome to `augment the training
data <https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html>`__,
but this is not a requirement.

.. code:: ipython3

    ### TODO: Train the model.
    from keras.callbacks import ModelCheckpoint
    
    
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5', 
                                   verbose=1, save_best_only=True)
    
    VGG19_model.fit(train_VGG19, train_targets, 
              validation_data=(valid_VGG19, valid_targets),
              epochs=50, batch_size=20, callbacks=[checkpointer], verbose=1)


.. parsed-literal::

    Train on 6680 samples, validate on 835 samples
    Epoch 1/50
    6680/6680 [==============================] - 4s 667us/step - loss: 11.5630 - acc: 0.1446 - val_loss: 9.9686 - val_acc: 0.2216
    
    Epoch 00001: val_loss improved from inf to 9.96857, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 2/50
    6680/6680 [==============================] - 2s 282us/step - loss: 9.2374 - acc: 0.3189 - val_loss: 9.1299 - val_acc: 0.3257
    
    Epoch 00002: val_loss improved from 9.96857 to 9.12990, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 3/50
    6680/6680 [==============================] - 2s 288us/step - loss: 8.5379 - acc: 0.3954 - val_loss: 8.6997 - val_acc: 0.3689
    
    Epoch 00003: val_loss improved from 9.12990 to 8.69966, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 4/50
    6680/6680 [==============================] - 2s 294us/step - loss: 8.0877 - acc: 0.4361 - val_loss: 8.3278 - val_acc: 0.3701
    
    Epoch 00004: val_loss improved from 8.69966 to 8.32785, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 5/50
    6680/6680 [==============================] - 2s 284us/step - loss: 7.6138 - acc: 0.4716 - val_loss: 8.0941 - val_acc: 0.3856
    
    Epoch 00005: val_loss improved from 8.32785 to 8.09408, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 6/50
    6680/6680 [==============================] - 2s 301us/step - loss: 7.3879 - acc: 0.4967 - val_loss: 7.9559 - val_acc: 0.4012
    
    Epoch 00006: val_loss improved from 8.09408 to 7.95588, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 7/50
    6680/6680 [==============================] - 2s 274us/step - loss: 7.1171 - acc: 0.5175 - val_loss: 7.7348 - val_acc: 0.4060
    
    Epoch 00007: val_loss improved from 7.95588 to 7.73477, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 8/50
    6680/6680 [==============================] - 2s 273us/step - loss: 6.8114 - acc: 0.5386 - val_loss: 7.4557 - val_acc: 0.4455
    
    Epoch 00008: val_loss improved from 7.73477 to 7.45569, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 9/50
    6680/6680 [==============================] - 2s 266us/step - loss: 6.7039 - acc: 0.5582 - val_loss: 7.4967 - val_acc: 0.4407
    
    Epoch 00009: val_loss did not improve
    Epoch 10/50
    6680/6680 [==============================] - 2s 276us/step - loss: 6.5529 - acc: 0.5642 - val_loss: 7.1095 - val_acc: 0.4659
    
    Epoch 00010: val_loss improved from 7.45569 to 7.10950, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 11/50
    6680/6680 [==============================] - 2s 263us/step - loss: 6.1888 - acc: 0.5889 - val_loss: 7.0672 - val_acc: 0.4611
    
    Epoch 00011: val_loss improved from 7.10950 to 7.06724, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 12/50
    6680/6680 [==============================] - 2s 259us/step - loss: 6.0960 - acc: 0.6037 - val_loss: 6.9489 - val_acc: 0.4850
    
    Epoch 00012: val_loss improved from 7.06724 to 6.94893, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 13/50
    6680/6680 [==============================] - 2s 267us/step - loss: 6.0370 - acc: 0.6123 - val_loss: 6.9932 - val_acc: 0.4790
    
    Epoch 00013: val_loss did not improve
    Epoch 14/50
    6680/6680 [==============================] - 2s 260us/step - loss: 6.0169 - acc: 0.6151 - val_loss: 6.9966 - val_acc: 0.4874
    
    Epoch 00014: val_loss did not improve
    Epoch 15/50
    6680/6680 [==============================] - 2s 268us/step - loss: 5.9359 - acc: 0.6196 - val_loss: 6.7697 - val_acc: 0.4946
    
    Epoch 00015: val_loss improved from 6.94893 to 6.76966, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 16/50
    6680/6680 [==============================] - 2s 271us/step - loss: 5.8533 - acc: 0.6296 - val_loss: 6.7608 - val_acc: 0.5018
    
    Epoch 00016: val_loss improved from 6.76966 to 6.76083, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 17/50
    6680/6680 [==============================] - 2s 268us/step - loss: 5.8416 - acc: 0.6316 - val_loss: 6.8525 - val_acc: 0.4934
    
    Epoch 00017: val_loss did not improve
    Epoch 18/50
    6680/6680 [==============================] - 2s 297us/step - loss: 5.8010 - acc: 0.6320 - val_loss: 6.7257 - val_acc: 0.5090
    
    Epoch 00018: val_loss improved from 6.76083 to 6.72568, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 19/50
    6680/6680 [==============================] - 2s 315us/step - loss: 5.7217 - acc: 0.6392 - val_loss: 6.7369 - val_acc: 0.5126
    
    Epoch 00019: val_loss did not improve
    Epoch 20/50
    6680/6680 [==============================] - 2s 260us/step - loss: 5.7033 - acc: 0.6430 - val_loss: 6.6891 - val_acc: 0.5054
    
    Epoch 00020: val_loss improved from 6.72568 to 6.68913, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 21/50
    6680/6680 [==============================] - 2s 259us/step - loss: 5.6657 - acc: 0.6422 - val_loss: 6.6763 - val_acc: 0.5114
    
    Epoch 00021: val_loss improved from 6.68913 to 6.67625, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 22/50
    6680/6680 [==============================] - 2s 259us/step - loss: 5.5234 - acc: 0.6499 - val_loss: 6.5266 - val_acc: 0.5234
    
    Epoch 00022: val_loss improved from 6.67625 to 6.52658, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 23/50
    6680/6680 [==============================] - 2s 261us/step - loss: 5.4217 - acc: 0.6534 - val_loss: 6.4960 - val_acc: 0.5126
    
    Epoch 00023: val_loss improved from 6.52658 to 6.49603, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 24/50
    6680/6680 [==============================] - 2s 259us/step - loss: 5.2094 - acc: 0.6663 - val_loss: 6.3927 - val_acc: 0.5281
    
    Epoch 00024: val_loss improved from 6.49603 to 6.39275, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 25/50
    6680/6680 [==============================] - 2s 267us/step - loss: 5.1315 - acc: 0.6735 - val_loss: 6.2401 - val_acc: 0.5329
    
    Epoch 00025: val_loss improved from 6.39275 to 6.24007, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 26/50
    6680/6680 [==============================] - 2s 268us/step - loss: 5.0499 - acc: 0.6811 - val_loss: 6.2274 - val_acc: 0.5497
    
    Epoch 00026: val_loss improved from 6.24007 to 6.22744, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 27/50
    6680/6680 [==============================] - 2s 262us/step - loss: 4.9889 - acc: 0.6807 - val_loss: 6.1140 - val_acc: 0.5461
    
    Epoch 00027: val_loss improved from 6.22744 to 6.11403, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 28/50
    6680/6680 [==============================] - 2s 259us/step - loss: 4.8667 - acc: 0.6859 - val_loss: 6.0715 - val_acc: 0.5461
    
    Epoch 00028: val_loss improved from 6.11403 to 6.07146, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 29/50
    6680/6680 [==============================] - 2s 260us/step - loss: 4.7974 - acc: 0.6964 - val_loss: 6.1235 - val_acc: 0.5425
    
    Epoch 00029: val_loss did not improve
    Epoch 30/50
    6680/6680 [==============================] - 2s 262us/step - loss: 4.7865 - acc: 0.6984 - val_loss: 6.0475 - val_acc: 0.5401
    
    Epoch 00030: val_loss improved from 6.07146 to 6.04748, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 31/50
    6680/6680 [==============================] - 2s 263us/step - loss: 4.7810 - acc: 0.6997 - val_loss: 6.0167 - val_acc: 0.5569
    
    Epoch 00031: val_loss improved from 6.04748 to 6.01666, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 32/50
    6680/6680 [==============================] - 2s 263us/step - loss: 4.7468 - acc: 0.6996 - val_loss: 6.0238 - val_acc: 0.5521
    
    Epoch 00032: val_loss did not improve
    Epoch 33/50
    6680/6680 [==============================] - 2s 260us/step - loss: 4.7178 - acc: 0.7019 - val_loss: 5.9533 - val_acc: 0.5617
    
    Epoch 00033: val_loss improved from 6.01666 to 5.95332, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 34/50
    6680/6680 [==============================] - 2s 261us/step - loss: 4.6969 - acc: 0.7058 - val_loss: 5.9170 - val_acc: 0.5701
    
    Epoch 00034: val_loss improved from 5.95332 to 5.91702, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 35/50
    6680/6680 [==============================] - 2s 263us/step - loss: 4.6048 - acc: 0.7082 - val_loss: 5.8234 - val_acc: 0.5665
    
    Epoch 00035: val_loss improved from 5.91702 to 5.82344, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 36/50
    6680/6680 [==============================] - 2s 264us/step - loss: 4.4719 - acc: 0.7148 - val_loss: 5.7297 - val_acc: 0.5713
    
    Epoch 00036: val_loss improved from 5.82344 to 5.72971, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 37/50
    6680/6680 [==============================] - 2s 261us/step - loss: 4.4100 - acc: 0.7220 - val_loss: 5.7383 - val_acc: 0.5749
    
    Epoch 00037: val_loss did not improve
    Epoch 38/50
    6680/6680 [==============================] - 2s 280us/step - loss: 4.3926 - acc: 0.7237 - val_loss: 5.6755 - val_acc: 0.5796
    
    Epoch 00038: val_loss improved from 5.72971 to 5.67551, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 39/50
    6680/6680 [==============================] - 2s 278us/step - loss: 4.3897 - acc: 0.7256 - val_loss: 5.6920 - val_acc: 0.5832
    
    Epoch 00039: val_loss did not improve
    Epoch 40/50
    6680/6680 [==============================] - 2s 277us/step - loss: 4.3854 - acc: 0.7260 - val_loss: 5.5741 - val_acc: 0.5940
    
    Epoch 00040: val_loss improved from 5.67551 to 5.57412, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 41/50
    6680/6680 [==============================] - 2s 278us/step - loss: 4.3690 - acc: 0.7265 - val_loss: 5.6893 - val_acc: 0.5713
    
    Epoch 00041: val_loss did not improve
    Epoch 42/50
    6680/6680 [==============================] - 2s 288us/step - loss: 4.3175 - acc: 0.7272 - val_loss: 5.9479 - val_acc: 0.5557
    
    Epoch 00042: val_loss did not improve
    Epoch 43/50
    6680/6680 [==============================] - 2s 289us/step - loss: 4.2514 - acc: 0.7269 - val_loss: 5.5388 - val_acc: 0.5856
    
    Epoch 00043: val_loss improved from 5.57412 to 5.53876, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 44/50
    6680/6680 [==============================] - 2s 279us/step - loss: 4.2035 - acc: 0.7341 - val_loss: 5.5546 - val_acc: 0.5844
    
    Epoch 00044: val_loss did not improve
    Epoch 45/50
    6680/6680 [==============================] - 2s 286us/step - loss: 4.1879 - acc: 0.7361 - val_loss: 5.5151 - val_acc: 0.5856
    
    Epoch 00045: val_loss improved from 5.53876 to 5.51514, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 46/50
    6680/6680 [==============================] - 2s 277us/step - loss: 4.1801 - acc: 0.7376 - val_loss: 5.5085 - val_acc: 0.5940
    
    Epoch 00046: val_loss improved from 5.51514 to 5.50847, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 47/50
    6680/6680 [==============================] - 2s 280us/step - loss: 4.1743 - acc: 0.7383 - val_loss: 5.4968 - val_acc: 0.5952
    
    Epoch 00047: val_loss improved from 5.50847 to 5.49684, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 48/50
    6680/6680 [==============================] - 2s 317us/step - loss: 4.1039 - acc: 0.7383 - val_loss: 5.3527 - val_acc: 0.5940
    
    Epoch 00048: val_loss improved from 5.49684 to 5.35266, saving model to saved_models/weights.best.VGG19.hdf5
    Epoch 49/50
    6680/6680 [==============================] - 2s 327us/step - loss: 3.9487 - acc: 0.7460 - val_loss: 5.4475 - val_acc: 0.5868
    
    Epoch 00049: val_loss did not improve
    Epoch 50/50
    6680/6680 [==============================] - 2s 313us/step - loss: 3.8945 - acc: 0.7513 - val_loss: 5.3489 - val_acc: 0.6000
    
    Epoch 00050: val_loss improved from 5.35266 to 5.34892, saving model to saved_models/weights.best.VGG19.hdf5




.. parsed-literal::

    <keras.callbacks.History at 0x1a343c0160>



(IMPLEMENTATION) Load the Model with the Best Validation Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ### TODO: Load the model weights with the best validation loss.
    VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')

(IMPLEMENTATION) Test the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try out your model on the test dataset of dog images. Ensure that your
test accuracy is greater than 60%.

.. code:: ipython3

    ### TODO: Calculate classification accuracy on the test dataset.
    import numpy as np
    # get index of predicted dog breed for each image in test set
    VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]
    
    # report test accuracy
    test_accuracy = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


.. parsed-literal::

    Test accuracy: 58.4928%


(IMPLEMENTATION) Predict Dog Breed with the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write a function that takes an image path as input and returns the dog
breed (``Affenpinscher``, ``Afghan_hound``, etc) that is predicted by
your model.

Similar to the analogous function in Step 5, your function should have
three steps: 1. Extract the bottleneck features corresponding to the
chosen CNN model. 2. Supply the bottleneck features as input to the
model to return the predicted vector. Note that the argmax of this
prediction vector gives the index of the predicted dog breed. 3. Use the
``dog_names`` array defined in Step 0 of this notebook to return the
corresponding breed.

The functions to extract the bottleneck features can be found in
``extract_bottleneck_features.py``, and they have been imported in an
earlier code cell. To obtain the bottleneck features corresponding to
your chosen CNN architecture, you need to use the function

::

    extract_{network}

where ``{network}``, in the above filename, should be one of ``VGG19``,
``Resnet50``, ``InceptionV3``, or ``Xception``.

.. code:: ipython3

    from sklearn.datasets import load_files       
    from keras.utils import np_utils
    import numpy as np
    from glob import glob
    
    # define function to load train, test, and validation datasets
    def load_dataset(path):
        data = load_files(path)
        dog_files = np.array(data['filenames'])
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets
    
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('dogImages/train')
    valid_files, valid_targets = load_dataset('dogImages/valid')
    test_files, test_targets = load_dataset('dogImages/test')
    
    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
    
    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))


.. parsed-literal::

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


.. code:: ipython3

    ### TODO: Write a function that takes a path to an image as input
    ### and returns the dog breed that is predicted by the model.
    from extract_bottleneck_features import *
    from IPython.core.display import Image, display
    
    def VGG19_predict_breed(img_path):
        # extract bottleneck features
        bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = VGG19_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]
        

--------------

 ## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first
determines whether the image contains a human, dog, or neither. Then, -
if a **dog** is detected in the image, return the predicted breed. - if
a **human** is detected in the image, return the resembling dog breed. -
if **neither** is detected in the image, provide output that indicates
an error.

You are welcome to write your own functions for detecting humans and
dogs in images, but feel free to use the ``face_detector`` and
``dog_detector`` functions developed above. You are **required** to use
your CNN from Step 5 to predict dog breed.

Some sample output for our algorithm is provided below, but feel free to
design your own user experience!

.. figure:: images/sample_human_output.png
   :alt: Sample Human Output

   Sample Human Output

(IMPLEMENTATION) Write your Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def dog_breed_algorithm(img_path):
        display(Image(img_path,width=200,height=200))
        if dog_detector(img_path) == 1:
            print("This is believed to be a dog. Its predicted breed is: ")
            return VGG19_predict_breed(img_path)
        elif face_detector(img_path) == 1:
            print("This is believed to be a human. Its predicted breed is: ")
            return VGG19_predict_breed(img_path)
        else:
            return print("That's not a dog nor an human.")




.. image:: output_71_0.jpeg
   :width: 200px
   :height: 200px


.. parsed-literal::

    This is believed to be a dog. Its predicted breed is: 
    Poodle


.. code:: ipython3

    ### TODO: Write your algorithm.Z
    ### Feel free to use as many code cells as needed.

--------------

 ## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin! What kind
of dog does the algorithm think that **you** look like? If you have a
dog, does it predict your dog's breed accurately? If you have a cat,
does it mistakenly think that your cat is a dog?

(IMPLEMENTATION) Test Your Algorithm on Sample Images!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test your algorithm at least six images on your computer. Feel free to
use any images you like. Use at least two human and two dog images.

**Question 6:** Is the output better than you expected :) ? Or worse :(
? Provide at least three possible points of improvement for your
algorithm.

**Answer:**

.. code:: ipython3

    ## TODO: Execute your algorithm from Step 6 on
    ## at least 6 images on your computer.
    ## Feel free to use as many code cells as needed.
    print(dog_breed_algorithm('/Users/Christophe/dog-project/images/American_water_spaniel_00648.jpg'))
    print(dog_breed_algorithm('/Users/Christophe/dog-project/images/Brittany_02625.jpg'))
    print(dog_breed_algorithm('/Users/Christophe/dog-project/images/Curly-coated_retriever_03896.jpg'))
    print(dog_breed_algorithm('/Users/Christophe/dog-project/images/Labrador_retriever_06449.jpg'))
    print(dog_breed_algorithm('/Users/Christophe/dog-project/images/sample_human_output.png'))
    print(dog_breed_algorithm('/Users/Christophe/dog-project/images/sample_cnn.png'))




.. image:: output_74_0.jpeg
   :width: 200px
   :height: 200px


.. parsed-literal::

    This is believed to be a dog. Its predicted breed is: 
    Poodle



.. image:: output_74_2.jpeg
   :width: 200px
   :height: 200px


.. parsed-literal::

    This is believed to be a dog. Its predicted breed is: 
    Brittany



.. image:: output_74_4.jpeg
   :width: 200px
   :height: 200px


.. parsed-literal::

    This is believed to be a dog. Its predicted breed is: 
    Curly-coated_retriever



.. image:: output_74_6.jpeg
   :width: 200px
   :height: 200px


.. parsed-literal::

    This is believed to be a dog. Its predicted breed is: 
    Labrador_retriever



.. image:: output_74_8.png
   :width: 200px
   :height: 200px


.. parsed-literal::

    This is believed to be a human. Its predicted breed is: 
    English_toy_spaniel



.. image:: output_74_10.png
   :width: 200px
   :height: 200px


.. parsed-literal::

    That's not a dog nor an human.
    None

