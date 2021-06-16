## Recommendations and Limitations

Although our project has demonstrated that machine learning can be used to detect lipohypertrophy, there are some key limitations that should be addressed before it is used in the clinical setting. These key limitations are as follows:

1. Scarcity of data - We had merely 353 images to develop our model which is a small dataset in the realm of deep learning. The scarcity of our dataset caused variability in different runs of the experiments conducted. We have also noticed that our model’s performance is sensitive to data splitting (i.e. which images are in the train, validation and test folders). 

2. Limited resources - Limited computing power has been a recurrent challenge in this project and has made it difficult to experiment with tuning all the parameters of our model concurrently. 

3. Lack of Oversight - We believe that there should be an auditing process involved to ensure that our machine learning model does not propagate any biases that could cause harm to specific patient populations. It would also be useful to have other independent radiologists label the dataset in order to draw comparisons between physicians’ and algorithms’ performance.

To address the limitations outlined above, our team has made the following recommendations: 

1.  Addition of more labelled images – In line with the size of the dataset in similar studies (Xiao et al., 2018; Cheng & Malhi, 2017), we recommend increasing the dataset to at least 1000 images.

2.  Increasing computational resources - we recommend obtaining funding for additional computing resources for both the development and deployment of future models.

3. Add additional functionality such as native cropping.
