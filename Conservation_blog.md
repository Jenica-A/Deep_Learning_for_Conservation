## Deep Learning for Conservation
## Classifying Drone Images from Protected Lands: Deep Learning For Conservation

Deep learning Convolution Neural Networks (CNNs) are deployed to automate image analyses for such tasks as face recognition, object detection, security, and more. But is the success of a model dependent on high quality images, which may be expensive in terms of camera equipment, data storage, data management and high resolution image transmission? Can application of this Machine Learning method be utilized by organizations with smaller budgets, say non-profits with conservation in mind? authorities to respond to fires, illegal hunting or illegal land use. Government protection has not been considered sufficient up to this point (source) and an automated response would enable appropriate emergency response to be deployed as quickly as possible.

The goal of my research was to demonstrate the capabilities of CNNs to classify low quality, small-sized, aerial images for environmental monitoring. This specific project classified images from the Tehuacan-Cuicatlan Biosphere Reserve in Mexico. The reserve is a protected area that is proximal to areas where illegal mining, hunting and logging has historically occurred. In-person surveillance has proven costly and, according to Reuters and BBC articles, susceptible to corruption in other similar areas. 

My project analyzed images collected by unmanned air vehicles (drones) as either containing columnar cacti, a plant that is widely dispersed across the preserve, or not. The application of the work is automating surveillance, as part of a growing effort to develop solutions and ultimately detect activity. 
The dataset included 21,500 images from Kaggle's "Cactus Aerial Photos" dataset; 66% were positive class (containing cacti), 33% were the negative class (lacking cacti). Images range from 325 bytes to 61KB in size, and consisting of two categories--about 16,100 cactus images and 5,364 non-cactus images. My model split the data into 80% training data images and 20% testing data images.

Jupyter notebooks and python were my primary tools. I used tensorflow and keras to construct models. I used Matplotlib for visualizations. I initially applied transfer learning via the MobileNet model, which was trained on the ImageNet database. These results were poor and were used for comparative reasons as I built and refined a custom. Transfer learning was not utilized in the final model.

I constructed a keras model to perform the analysis. Many model architectures were tested throughout this project. Only the final model is described here. The model was based on the keras deep learning tutorial found here. A few trials and modifications resulted in the final model consisting of a rescaling layer (a preprocessing layer which multiplied input values by 1./255 to be between 0 and 1), eleven BatchNormalization layers (which applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1 (source), and which results in a model that requires fewer epochs to converge), nine SeparableConv2D layers for the deeper convolution layers (which better handles images where uncertainty exists around pixel similarity to neighboring pixels), one MaxPooling2D layer, one GlobalAveragePooling layer (which both reduce the dimensions of the feature map), and a Dropout layer (which helps prevent model overfitting).
The final model included a total of 2,773,913 trainable parameters and 8,736 non-trainable parameters, relied on ReLU activation and "Adam" optimization. The final result showed 99.2 accuracy, and 0.02 binary cross-entropy loss on the test data.

**Final training scores: loss: 0.0212 - accuracy: 0.9930**
**Final validation scores val_loss: 0.0223 - val_accuracy: 0.9921**

### Communication
See accompanying slide deck (for oral presentation and defense on July 13, 2022), and jupyter notebooks. All materials are available in my personal github account as well.