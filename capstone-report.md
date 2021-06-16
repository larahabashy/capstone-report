# Capstone Project Report
## MDS 2021 Capstone Project with the Gerontology Diabetes Research Lab (GDRL)
By: Ela Bandari, Lara Habashy, Peter Yang and Javairia Raza

# Executive Summary
Subclinical lipohypertrophy is traditionally evaluated based on visual inspection or palpation. Recent work has shown that lipohypertrophy may be detected by ultrasound imaging (Kapeluto et al., 2018). However, the criteria used to classify lipohypertrophy using ultrasound imaging is only familiar to and implemented by a small group of physicians (Madden, 2021). In an effort to improve the accessibility and efficiency of this method of detection, we have developed a supervised machine learning model to detect lipohypertrophy in ultrasound images. 

To develop our machine learning model, we tested a variety of image augmentation techniques and ultimately decided on augmenting our dataset by adding random flipping, contrast and brightness. We then fit a variety of pre-existing model architectures trained on thousands of images to our augmented dataset. We optimized the parameters by which the model learned and then carefully examined the different models’ performance and fine tuned them to our dataset before selecting the best performing model. Our final model accurately classified 76% of unseen test data. We have made our model accessible to potential users via a [web interface](https://share.streamlit.io/xudongyang2/lipo_deployment/demo_v2.py). Our work has demonstrated the potential that supervised machine learning holds in accurately detecting lipohypertrophy; however, the scarcity of the ultrasound images has been a contributor to the model’s shortcomings. We have outlined a set of recommendations for improving this project; the most notable of which is re-fitting the model using a larger dataset.

## Introduction
Subclinical lipohypertrophy is a common complication for diabetic patients who inject insulin. It is defined as the growth of fat cells and fibrous tissue in the deepest layer of the skin following repeated insulin injections in the same area. It is critical that insulin is not injected into areas of lipohypertrophy as it reduces the effectiveness of the insulin. Fortunately, research by Kapeluto et al. (2018)  has found ultrasound imaging techniques are more accurate in finding these masses than a physical examination of the body by a healthcare professional. But, currently, the criteria to classify lipohypertrophy using ultrasound imaging is only implemented by a small group of physicians. To expand the usability of this criteria to a larger set of healthcare professionals, the capstone partner is interested in seeing if we can leverage supervised machine learning techniques to accurately classify the presence of lipohypertrophy given an ultrasound image.

[insert positive and negative image examples here]

Our current dataset includes 218 negative images (no lipohypertrophy present) and 135 positive images (lipohypertrophy present) which is typically considered to be a very small dataset for a deep learning model. Our specific data science objectives for this project include:

[insert df of counts of positive and negative]

 1) Use the provided dataset to develop and evaluate the efficacy of an image classification model.
 2) Deploy the model for a non-technical audience
 
## Literature Review

Prediction of masses in ultrasound images using machine learning techniques has been an ongoing effort in clinical practice for the past few decades. To assist physicians in diagnosing disease, many scholars have implemented techniques such as regression, decision trees, Naive Bayesian classifiers, and neural networks on patients' ultrasound imaging data. (ref) 
We have found two studies (Chaio et al., 2019 & Yasaka et al., 2017) that showed success using CNNs with ~85% accuracy. 

Furthermore, due to the scarcity of ultrasound images, recent research has delved into various image augmentation approaches. We have found that both traditional and complex transformations such as using GANs are being used to generate images(ref), enlarging the dataset used for training which naturally improves the performance. Many other studies such as Esteva et al., 2017;Loey et al 2020 confirmed that minimal transformations such as simple flips, achieved higher prediction accuracy for their dataset. 

We also found that transfer learning architectures are crucial for yielding reliable results as any given dataset from one application is likely too small. This study built a CNN using DenseNet models for the prediction of breast cancer masses and achieved a test accuracy of 91%. Zhang, H., Han, L., Chen, K. (2020) also studying the diagnosis of breast cancer tumours in ultrasound images found that when compared with other architectures such as InceptionV3, VGG16, ResNet50, and VGG19, InceptionV3 performs the best. Prediction competitions on Kaggle have proven successful with architectures such as VGG16 and Resnet, predicting masses in ultrasound images with 70-80% accuracy. [add ref-Ela] 
For the object detection piece, we considered RCNNs but after some research we saw YOLO performs well in Kaggle competitions and is generally faster than RCNNS so we went with the YOLO model (reference).

[Add table: ref, author, technique, accuracy]

## Data Science Methods 

3.1 Data Preparation

For image transformations, we explored the pytorch and the albumentations library as they proved to improve accuracy (Al-Dhabyani et al., 2019). We attempted to leverage a machine learning tool called autoalbument from the latter library that finds the best transformations for you. However, following several runs, the accuracy did not improve from baseline. We suspect it may be that the transformations were too complex for the model to learn anything significant. Using simpler transformations to augment the data was complemented by results found in our literature review(Esteva et al., 2017;Loey et al., 2020). A variety of classic transformations were tested and the model’s performance on these augmented dataset were documented <insert link to manual notebook>. The transformations that led to the best performance were adding random vertical and horizontal flipping with a probability of 50% along with random brightness and contrast. 

3.2 Model Development 

The augmented data is then used to train a CNN model using transfer learning. There are many popular architectures for transfer learning models that have proven successful as discussed in our literature review. As such, we investigated the following structures: VGG16, ResNet, DenseNet and InceptionV3. Each of the models were incorporated with our small dataset, trained in separate experiments utilizing techniques to optimize the parameters of the model to maximize its ability to learn. To compare the performance of the different architectures, the team considered the accuracy and recall scores of the models when tested on our test set. 

[insert table of results that Javairia made]

The model that utilizes the densenet architecture appears to be the best performing model for our dataset. Furthermore, we conducted a manual analysis of tricky negative and positive images to test model performance using an interactive interface. This application allowed us to compare how confident the models were when misclassifying cases and showcased how well DenseNet was doing. 
In an effort to reduce the high confidence levels for misclassifications, a problem known as overfitting, we considered adding dropout layers in our CNN structure. In this case, overfitting occurs as a result of having trained our models on such a small training set. Another way to reduce overfitting is called an ensemble. However, an ensemble would not be feasible as it requires lots of resources such as enough CPU to train the models.. Although the dropout layers did manage to reduce overfitting in the models, the overall reduced accuracy was not significant enough to implement those layers in our optimal model. Also, further discussion with our capstone partner, we found that nurses are likely to avoid areas where the model is not confident altogether.To see this exploration, click here.
Furthermore, as our capstone partner was far more interested in reducing false negatives, areas where insulin should not be injected, we investigated various techniques to optimize recall. Positive weight (pos_weight) is one argument of the loss function used in our CNN model which, if greater than 1, prioritizes the positive examples more such that there is a heavier penalization (loss) on labelling the positive lipohypertrophy examples incorrectly. However, this method was not implemented in our final model as it proved to be unstable. After conducting a few experiments, we found high variance in the results. To see this exploration, click here.
Finally, the choice of our optimal model as DenseNet was further motivated by its size and compatibility with our data product mentioned in the next section. <insert size comparison Javairia made> Our final objective was to implement object detection into our pipeline, which allows us to identify the exact location of lipohypertrophy on a given image. We’ve seen amazing results using the YOLOv5 framework - which is one of the best available frameworks right now. (ref) First, using the annotated ultrasound images, the team created bounding boxes around the exact location of the lipohypertrophy masses in the positive training images. Next, using the YOLOv5 framework, the Yolov5m model was trained for approximately 200 epochs with an image size of 300 and a batch size of 8. The team also experimented with different training parameters to find the optimal ones and found the former specifics optimal.

Result: In testing, the YOLOv5 framework was quite confident in the predictions of bounding boxes around exact locations of many of the images (numbers?). 

[Example of Object Detection]

## Data Product and Results 

[Web app screenshot]

The first deliverable of the data product is the source code including an environmental file (.yml), command-line enabled python scripts to perform train-validation-test split on image folders, modeling, evaluation, and deployment, and a Makefile that automates the entire process. The team expects our partner to use the source code to refit the lipohypertrophy classification model as more ultrasound images become available. The python scripts are built with optional arguments for model hyperparameters, which makes this a flexible tool to work with. The Makefile renders the process automatic and makes model refitting effortless. However, the source code requires a certain degree of python and command line interface knowledge.

The second deliverable is a web application for our partner to interact with the lipohypertrophy classification and object detection models. The app is deployed on Streamlit share and Heroku, free-service platforms. The team expects our partner to use this app to upload one or multiple cropped ultrasound images, select one of them with the select box, and get an instance prediction of lipohypertrophy presence and prediction confidence. Finally, if it is a positive prediction, a red box will appear on the image to indicate the proposed lipohypertrophy area. This web app has good accessibility as our partner can interact with it without IT clearance issues, thus making it easy to work with. However, the final models are hosted on Streamlit share after the team put in a special request to get more resources. After the Streamlit granted additional resources, it proved to be much faster than Heroku. Future (larger) models could be hosted on cloud based servers since they are more flexible and secure. The team has also considered creating a desktop app for lipohypertrophy classification model. A desktop app allows our partner to interact with the model without the need of internet access, and has low maintenance cost. However, downloading it from the internet requires internal IT clearance and this method was deemed less preferable by our partner.

## Conclusion

In this project, we aimed to investigate whether supervised machine learning techniques could be leveraged to detect the presence of lipohypertrophy in ultrasound images. Our trained models have demonstrated that this is indeed a possibility as they are accurately predicting the presence of lipohypertrophy on 76% of previously unseen data. Furthermore, our team has developed two data products. The data products are: 1) Well-documented source code and an automated machine learning pipeline, and 2) An interactive web application. The source code will allow future researchers and developers to inspect and integrate all aspects of the model. The Makefile included with the project makes it seamless to update the model with an expanded dataset. The web application allows healthcare providers to easily interact with our machine learning model to discover which sites are safe for insulin injection and which sites should be avoided. Although our project has demonstrated that machine learning can be used to detect lipohypertrophy, there are some key limitations and recommendations that should be addressed before it is used in the clinical setting (see Appendix). 


## References
[insert here]

