#!/usr/bin/env python
# coding: utf-8

# # Capstone Project Report 
# ## *MDS 2021 Capstone Project with the Gerontology Diabetes Research Lab (GDRL)*

# ## Executive Summary

# *A brief and high level summary of the project proposal.*

# > proposal: Subclinical lipohypertrophy is traditionally evaluated based on visual inspection or palpation. However, recent work has shown that lipohypertrophy may be detected by ultrasound imaging. The criteria used to classify lipohypertrophy using ultrasound imaging is only familiar to and implemented by a small group of physicians. In an effort to improve the accessibility and efficiency of this method of detection, our capstone partner has asked us to explore the possibility of using supervised machine learning to detect lipohypertrophy on ultrasound images. 
# 
# > In this project, we will be creating a convolutional neural network to detect the presence of lipohypertrophy in ultrasound images. We will be testing a variety of image preprocessing and transfer learning methodologies to select our finalized machine learning pipeline. Our proposed data product is a Python application that can intake new ultrasound images and make accurate predictions on the presence or absence of lipohypertrophy. 

# <br>

# ## Introduction

# *The introduction should start broad, introducing the question being asked/problem needing solving and why it is important. Any relevant information to understand the question/problem and its importance should be included. This may be similar to the problem statement of the proposal, but it has most likely evolved at least a little.*
# 
# *Over the course of the project, you’ve refined the problem statement into tangible objectives that can be directly assessed by data science techniques. Indicate these refined tangible objectives here, along with your reasoning for why these scientific objectives are useful for addressing the capstone partner’s needs. You might need to describe the available data first before outlining the scientific objectives.*

# > proposal: Lipohypertrophy is a common complication for diabetic patients who inject insulin {cite}`kapeluto2018ultrasound`. It is defined as the growth of fat cells and fibrous tissue with lowered vascularity in the skin following repeated trauma of insulin injection in the same area. Our focus is on subclinical hypertrophy which forms in the subcutaneous layer (the deepest layer of the skin) {cite}`boundless`. It is critical that insulin is not injected into areas of lipohypertrophy as it reduces the effectiveness of the insulin such that patients are unable to manage their blood sugar levels and may require more insulin to achieve the same therapeutic benefits {cite}`kapeluto2018ultrasound`. Fortunately, research by Kapeluto et al. (2018) {cite}`kapeluto2018ultrasound` has found ultrasound imaging techniques are more accurate in finding these masses than a physical examination of the body by a healthcare professional. But, currently, the criteria to classify lipohypertrophy using ultrasound imaging is only implemented by a small group of physicians {cite}`madden_2021`. To expand the usability of this criteria to a larger set of healthcare professionals, the capstone partner is interested in seeing if we could leverage supervised machine learning techniques to accurately classify the presence of lipohypertrophy given an ultrasound image. 

# ## Data Science Methods

# *Next, describe the data science techniques you used to address the scientific objectives. In your discussion, you might want to include:*
# 
# *Pros and cons of using this method;
# Justification over other methods;
# An indication of a method that might work better, or improvements one could make to your techniques. Also indicate the difficulties associated with making these improvements, and why you didn’t implement the improvements.*

# ### Example table
# 
# |  | Positive | Negative |
# | --- | --- | --- |
# | Count | 135 | 128 |
# | Proportion | 51% | 49% |

# ### Example plots with figure captions

# In[1]:


fig = figure(figsize=(10, 8))
a=fig.add_subplot(1,1,1)
image = imread("proposal_feature_importance.PNG")
imshow(image)
axis('off')
txt = "Figure 2: Feature Importance. The lighter orange and yellow areas on the right represent areas that \n the baseline model thinks are important for detecting lipohypertrophy."
_ = figtext(0.5, 0.10, txt, horizontalalignment='center', fontsize=14)


# <br>

# ## Data product and results

# *Describe your data product and any relevant results obtained from it. For the description of the data product, you might want to include:*
# 
# *How you intend the partner to use the data product;
# Pros and cons of using this product/interface;
# Justification over other products/interfaces;
# An indication of a product/interface that might work better, or improvements one could make to your product/interface. Also indicate the difficulties associated with making these improvements, and why you didn’t implement the improvements.
# The description of your data product should not include an indication of how to technically use the data product, nor any other documentation to your product. Such “how-related” topics belong with the product itself (e.g., in the README). Rather, this final report should focus on the “why-related” topics to communicate the purpose of the product’s existence.*

# ## Conclusions and recommendations

# *Re-state the question being asked/problem needing solving and explain how your data product answers/solves it. Comment on how well your product answered their question/solved their problem, and discuss any limitations of the product in its ability of doing so. Describe recommendations for the partner to continue/improve the project in the future.*

# ## References

# <br>
