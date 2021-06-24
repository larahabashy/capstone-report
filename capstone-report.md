# Final Report
**MDS 2021 Capstone Project with the Gerontology Diabetes Research Lab (GDRL)**

By: Ela Bandari, Lara Habashy, Javairia Raza and Peter Yang 

## 1. Executive Summary
Subclinical lipohypertrophy is traditionally evaluated by visual inspection or palpation. Recent work has shown that lipohypertrophy may be detected by ultrasound imaging {cite:p}`kapeluto2018ultrasound`. However, the criteria used to classify lipohypertrophy using ultrasound imaging is only familiar to and implemented by a small group of physicians {cite:p}`madden_2021`. In an effort to improve the accessibility and efficiency of this method of detection, we have developed a supervised machine learning model to detect lipohypertrophy in ultrasound images. 

We developed an optimal machine learning model through an iterative process of data augmentation, hyperparameter tuning, and architecture selection. The final optimal model accurately classified 76% of unseen test data. We tested different image augmentation techniques and ultimately decided on adding random flipping, contrast and brightness. We then fit a variety of pre-existing model architectures trained on thousands of images to our augmented dataset. We optimized the parameters by which the model learned and then carefully examined the different models’ performance and fine tuned them to our dataset before selecting the best performing model. We have made our best performing model accessible to potential users via a [web interface](https://share.streamlit.io/xudongyang2/lipo_deployment/demo_v2.py). Our work has demonstrated the potential that supervised machine learning holds in accurately detecting lipohypertrophy; however, the scarcity of the ultrasound images has been a contributor to the model’s shortcomings. We have outlined a set of recommendations (see section {ref}`recommendations`) for improving this project; the most notable of which is re-fitting the model using a larger dataset.


## 2. Introduction
Subclinical lipohypertrophy is a common complication for diabetic patients who inject insulin. It is defined as the growth of fat cells and fibrous tissue in the deepest layer of the skin following repeated insulin injections in the same area. It is critical that insulin is not injected into areas of lipohypertrophy as it reduces the effectiveness of the insulin {cite:p}`kapeluto2018ultrasound`. However, recent research by {cite:t}`kapeluto2018ultrasound` has found ultrasound imaging techniques are more accurate in finding these masses than a physical examination of the body by a healthcare professional. Examples of these images are shown in {numref}`example` below. Unfortunately, the criteria to classify lipohypertrophy using ultrasound imaging is only implemented by a small group of physicians. To expand the usability of this criteria to a larger set of healthcare professionals, the capstone partner is interested in seeing if we can leverage supervised machine learning techniques to accurately classify the presence of lipohypertrophy given an ultrasound image.

```{figure} image/example.png
---
height: 500px
name: example
---
Some examples of images found in our dataset. Top row is negative (no lipohypertrophy present) and bottom row is positive (lipohypertrophy present) images where the yellow annotations indicate the exact area of the mass. 
```

Our current dataset includes 218 negative images (no lipohypertrophy present) and 135 positive images (lipohypertrophy present) as shown in {numref}`counts_df` below. Notably, this is considered to be a very small dataset for a deep learning model. 

Our specific data science objectives for this project include:

1. Use the provided dataset to develop and evaluate the efficacy of an image classification model. Some of our specific goals include:

    1. Data Augmentation: applying random image transformations as to expland the dataset (section {ref}`data-prep`)
  
    2. Transfer Learning: using a model that has been pre-trained on thousands of images and using it for our dataset (section {ref}`model-dev`)
  
    3. Optimization of Learning: figuring out what are the best parameters that will make the model most optimal for learning the data (section {ref}`model-dev`)
  
    4. Object Detection: training a model to detect the location of lipohypertrophy on an image (section {ref}`obj-detect`)
  
2. Deploy the model for a non-technical audience (section {ref}`data-product`). 


```{figure} image/counts_df.png
---
height: 100px
name: counts_df
---
The total number of positive and negative examples in our dataset.
```
## 3. Literature Review

Throughout the project, the team has consulted literature to gain insight and direction on current practices relevant to our problem and our dataset to guide and validate our decision-making. Below is a summary of the relevant findings.

### 3.1 Similar Work

Prediction of masses in ultrasound images using machine learning techniques has been an ongoing effort in clinical practice for the past few decades. To assist physicians in diagnosing disease, many scholars have implemented techniques such as regression, decision trees, Naive Bayesian classifiers, and neural networks on patients' ultrasound imaging data {cite}`huang2018machine`. Further, many studies involving ultrasound images have preprocessed the images to extract features. This study {cite}`chiao2019detection` shows that CNNs using ultrasound images perform better than radiomic models at predicting breast cancer tumours. Another recent study shows success in classifying liver masses into 1 out 5 categories with 84% accuracy using a CNN model {cite}`yasaka2018deep`.

Furthermore, recent research has delved into various complex image augmentation approaches such as using GANs to generate images {cite}`al2019deep`, enlarging the dataset used for training which naturally improves the performance. The study also found that traditional transformations managed to improve model performance. Many other studies such as {cite}`esteva2017, loey2020deep` confirmed that minimal transformations such as flipping the images, achieved higher prediction accuracy in their application. 

We also found that transfer learning architectures are crucial for yielding reliable results as any given patient's dataset is likely too small to construct a CNN architecture. This study by {cite:t}`sun2020deep` built a CNN using DenseNet models for the prediction of breast cancer masses and achieved a test accuracy of 91%. {cite:t}`zhang2020diagnostic` also studying the diagnosis of breast cancer tumours in ultrasound images found that when compared with other architectures such as InceptionV3, VGG16, ResNet50, and VGG19, InceptionV3 performs the best. Prediction competitions on Kaggle have proven successful with architectures such as VGG16 and Resnet, predicting masses in ultrasound images with 70-80% accuracy {cite}`kaggle2`. 

For the object detection piece, along with the YOLO framework, we considered RCNNs but research showed YOLO performs well in Kaggle competitions {cite}`kaggle1` and is generally faster than RCNNs. As such, we went with the YOLO model {cite}`gandhi_2018`.

## 4. Data Science Methods 

The general data science workflow for this project is shown in {numref}`workflow`. We discuss each step of the workflow in more detail in the sub-sections below.

```{figure} image/ds_workflow.png
---
height: 500px
name: workflow
---
General data science workflow for this capstone project. 
```
(data-prep)= 
### 4.1 Data Preparation

#### 4.1.1 Data Splitting

Before beginning any model training, the data was split into train, validation and test splits of 70%, 15% and 15% respectively. Here, we define baseline as a model that only predicts negative and has 62% accuracy. Our goal for this project was to develop a model that performed better than baseline. 

#### 4.1.2 Image Augmentation

Given the small size of the dataset, image augmentation techniques were used to expand the size of the training set and improve the model's generalizability. We explored the pytorch {cite:p}`NEURIPS2019_9015` and the albumentations libraries as they had proven successful in increasing accuracy in previous work {cite:p}`al2019deep`. We also attempted to leverage a machine learning tool called autoalbument from the latter library that searches for and selects the best transformations. However, following several runs, the accuracy did not improve from baseline. We suspect it may be that the transformations were too complex for the model to learn anything significant. Using simpler transformations to augment the data was complemented by results found in our literature review {cite:p}`esteva2017, loey2020deep`. A variety of classic transformations were tested and the model’s performance on these augmented datasets were documented [here](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/master/notebooks/manual-albumentation.ipynb). The transformations that led to the best performance were adding random vertical and horizontal flipping along with random brightness and contrast adjustment, with a probability of 50%, augmenting the images as seen below in {numref}`transformation`.

```{figure} image/transformed_image.png
---
height: 200px
name: transformation
---
Final image transformations included random vertical and horizontal flipping and random brightness and contrast adjustment.
```
(model-dev)=
### 4.2 Model Development 

The augmented data is then used to train a CNN model using transfer learning, a technique utilizing pre-trained models on thousands of images, which allows for training with our comparatively smaller dataset. Based on our literature review, the transfer learning architectures we chose to investigate were the following: VGG16, ResNet50, DenseNet169 and InceptionV3. Each of the models were incorporated with our small dataset, trained in separate experiments utilizing techniques to optimize the parameters of the model to maximize its ability to learn. To compare the performance of the different architectures, the team considered the accuracy and recall scores of the models when tested on our test set. When comparing the relative scores of accuracy shown in {numref}`acc_chart` and recall shown in {numref}`recall_chart`, DenseNet outperformed the other architectures. DenseNet has also proved successful in similar deep learning applications using small datasets such as <add ref>[](https://link.springer.com/article/10.1007/s10278-020-00357-7#auth-Heqing-Zhang), which we suspect is due to its ability to reduce the parameters in a model.


```{figure} image/model_acc_bar_chart.png
---
height: 150px
name: acc_chart
---
All four models were tested on a holdout sample to produce these accuracy results. 
```

```{figure} image/model_recall_bar_chart.png
---
height: 150px
name: recall_chart
---
All four models were tested on a holdout sample to produce these recall results. 
```
Furthermore, we conducted a [manual analysis](https://github.com/UBC-MDS/capstone-gdrl-lipo/blob/master/notebooks/model_decision_making.ipynb) of tricky negative and positive images to evaluate the models' performance using an interactive interface. This application, demonstrated below in {numref}`true_pos_all_wrong`, allowed us to compare how confident the models were when misclassifying images and showcased the strengths of the DenseNet architecture. 

```{figure} image/true_pos_all_wrong.png
---
height: 500px
name: true_pos_all_wrong
---
A manual inspection of tricky examples was conducted. Above, we have a true positive and although, all model are struggling, DenseNet is the least confident in its wrong prediction.  
```
In an effort to make the model more generalizable, we considered various techniques such as implementing dropout layers in our CNN structure, batch normalization and taking an ensemble approach. Furthermore, when identifying liphypertrophic masses, it is more detrimental for a true positive (areas where lipohypertrophy is present) to be falsely labelled as a negative. This is formally known as a false negative and thus, we also attempted to optimize recall, a score where a higher value indicates fewer false negatives. To find out more about these explorations and their outcomes, see [Appendix A - Documentation](../documentation.md).

Finally, the choice of our optimal model as DenseNet was further motivated by its relaively small computational size as seen in {numref}`size_df` below, and its compatibility with our data product, further discussed in {ref}`data-product`.

```{figure} image/size_df.png
---
height: 250px
name: size_df
---
Comparison of the file size of the different models revealed that DenseNet was the smallest model. 
```
(obj-detect)=
### 4.3 Object Detection

Our next objective was to implement object detection into our pipeline, giving us the ability to identify the exact location of lipohypertrophy on unseen test images. To implement object detection using a popular framework called YOLO {cite:p}`glenn_jocher_2020_4154370`, the team created bounding boxes around the location of the lipohypertrophy masses in the positive training images using the annotated ultrasound images as a guide. Next, using the YOLOv5 framework, the YOLOv5m model was trained for 200 epochs with an image size of 300 and a batch size of 8. The team experimented with different training parameters to find that the aforementioned training parameters produce optimal results. Below is a sample of those results identifying the exact location of lipohypertrophy with great confidence.

```{figure} image/object_detect_example.png
---
height: 500px
name: obj_det
---
Our final object detection model results on a test sample reveals promising results. The top row indicates the true location of lipohypertrophy and the bottom row indicates where the model thinks the lipohypertrophy is. The number on the red box indicates the model's confidence. 
```
(data-product)=
## 5. Data Product and Results 

Our project comprised of two key data products:

1. Source Code

2. Web application

We describe these products in detail below. 

The first deliverable data product is the source code including an environmental file (.yaml), command-line enabled python scripts to perform train-validation-test split on image folders, modeling, evaluation, and deployment, and a Makefile that automates the entire process. The team expects our partner to use the source code to refit the lipohypertrophy classification model as more ultrasound images become available. The python scripts are built with optional arguments for model hyperparameters, which makes this a flexible tool to work with. The Makefile renders the process automatic and makes the model refitting effortless. However, the source code requires a certain degree of python and command line interface knowledge.

The second deliverable is a web application as shown in {numref}`web_app` for our partner to interact with the lipohypertrophy classification and object detection models. A web application was an ideal choice compared to a local desktop app since the former does not require internal IT clearance, making it more ideal for our partner. The team expects our partner to use this app to upload one or multiple cropped ultrasound images, and get an instant prediction of lipohypertrophy presence along with the prediction confidence. If the prediction is positive, a red box will appear on the image to indicate the proposed lipohypertrophy area. The app is deployed on Streamlit share and Heroku, two free-service deployment platforms. The team sent a special request form to Streamlit to increase the allocated resources under Streamlit’s “good for the world” projects program. The request was approved and the app deployed on Streamlit share runs much faster in comparison to that deployed on Heroku; which now serves as a backup. Future (larger) models could be hosted on cloud based servers since they are more flexible and secure.

```{figure} image/demo_gif_v2.gif
---
height: 300px
name: web_app
---
Our [web application](https://share.streamlit.io/xudongyang2/lipo_deployment/demo_v2.py) on Streamlit can take multiple images and provides a final prediction and its confidence for a given image. If the model is considered positive, the model will also propose the location of the lipohypetrophy. 
```
(recommendations)=
## 6. Conclusions and recommendations

In this project, we aimed to investigate whether supervised machine learning techniques could be leveraged to detect the presence of lipohypertrophy in ultrasound images. Our trained models have demonstrated that this is indeed a possibility as they are accurately predicting the presence of lipohypertrophy on 76% of previously unseen data. Furthermore, our team has developed two data products. The data products are a well-documented source code and an automated machine learning pipeline and an interactive web application.

The open-source licensed source code will allow future researchers and developers to borrow from and build upon this work. The Makefile included with the project makes it seamless to update the model with an expanded dataset. The web application allows healthcare providers to easily interact with our machine learning model to discover which sites are safe for insulin injection and which sites should be avoided. 

Although our project has demonstrated that machine learning can be used to detect lipohypertrophy, there are some key limitations that should be addressed before it is used in the clinical setting. These key limitations are as follows:

1. Scarcity of data - We had merely 353 images to develop our model which is a small dataset in the realm of deep learning. The scarcity of our dataset caused variability in different runs of the experiments conducted. We have also noticed that our model’s performance is sensitive to data splitting (i.e. which images are in the training, validation and test folders). 

2. Limited resources - Limited computing power has been a recurrent challenge in this project and has made it difficult to experiment with tuning all the parameters of our model concurrently. 

3. Lack of oversight - We believe that there should be an auditing process involved to ensure that our machine learning model does not propagate any biases that could cause harm to specific patient populations. It would also be useful to have other independent radiologists label the dataset in order to draw comparisons between physicians’ and algorithms’ performance.

To address the limitations outlined above, our team has made the following recommendations: 

1.  Addition of more labelled images – in line with the size of the dataset in similar studies {cite:p}`xiao2018comparison,cheng2017transfer`, we recommend increasing the dataset to at least 1000 images and ideally up to several thousand images.

2.  Increasing computational resources - we recommend obtaining funding for additional computing resources for both the development and deployment of future models.

3. Adding additional functionality such as a cropping tool within the web interface.


## Bibliography

```{bibliography} references.bib
```