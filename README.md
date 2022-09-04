## Using Deep Learning Techniques to Monitor Protected Areas
Jenica Andersen<br>
Metis DSML, DLF module<br>
July 6, 2022

#### Abstract
The goal of this project was to demonstrate the capabilities of deep learning convolution neural networks, to classify low quality, small-sized aerial images for environmental monitoring. This specific case classified images of columnar cacti from the Tehuacan-Cuicatlan Biosphere Reserve in Mexico. The reserve is a protected area that is proximal to areas where illegal mining, hunting and logging has historically occurred in neighboring preserves. In-person surveillance has proven costly and susceptible to corruption by cartels. This demonstrates that images collected by unmanned air vehicles (drones), can be analyzed by deep learning neural networks to detect activity. This project demonstrates simply the technologically aided ability to discern images that contain or do not contain cacti as part of a growing effort to develop solutions. The dataset included 21,500 binary images, 66% were positive class, 33% were the negative class. I constructed a keras model to perform the analysis. The model included a total of 2,773,913 trainable parameters and 8,736 non-trainable parameters, relied on ReLU activation and "Adam" optimization. The final result showed 98.3 accuracy, 0.04 binary crossentropy loss. 

#### Design
The dataset originates from [Kaggle](https://www.kaggle.com/datasets/irvingvasquez/cactus-aerial-photos). It is the "Cactus Aerial Photos" dataset, and presents a two-class dataset: images containing cacti, or images not containing cacti. Classifying images accurately via machine learning models would enable the Mexican authorities to respond to fires, illegal hunting or illegal land use. Government protection has not been considered sufficient up to this point [(source)](https://jivg.org/research-projects/vigia/) and an automated response would enable appropriate emergency response to be deployed as quickly as possible. 

#### Data
The dataset contains 21,500 images ranging from 325 bytes to 61KB in size, and consisting of two categories--about 16,100 cactus images and 5,364 non-cactus images. My model split the data into 80% training data and 20% testing data. 

#### Tools
Jupyter notebooks and python were my primary tools. I used tensorflow and keras to construct models. I used Matplotlib for visualizations. Transfer learning was applied via the MobileNet model, which was trained on the ImageNet database; these results were unsatisfactory and were used for early results and comparative reasons, not for the final model.

#### Algorithms
*Final Models*

Many model architectures were tested throughout this project. Only the final model is described here. 
The model was based on the keras deep learning tutorial found [here](https://keras.io/examples/vision/image_classification_from_scratch/). A few trials and modifications resulted in the final model consisting of a rescaling layer (a preprocessing layer which multiplied input values by 1./255 to be between 0 and 1), eleven BatchNormalization layers (which applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1 [(source)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization), and which results in a model that requires fewer ephocs to converge), nine SeparableConv2D layers for the deeper convolution layers (which better handles images where uncertainty exists around pixel similarity to neighboring pixels), one MaxPooling2D layer, one GlobalAveragePooling layer (which both reduce the dimensions of the feature map), and a Dropout layer (which helps prevent model overfitting).

**Final training scores:** loss: 0.0407 - accuracy: 0.9867 

**Final validation scores** val_loss: 0.0473 - val_accuracy: 0.9828

#### Communication
See accompanying slide deck (for oral presentation, to be given on July 13, 2022), and jupyter notebooks. All materials will be posted to [my personal github account](https://github.com/Jenica-A/Deep_Learning_for_Conservation) as well. 
