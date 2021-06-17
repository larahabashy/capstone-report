# Javairia Raza
# 2021-06-06

# This simple script produces figures for the final report. It expects two folders to
# be downloaded from OneDrive: cropped_data and trained_models.

import os
import pandas as pd
import numpy as np
import altair as alt
from matplotlib.pyplot import figure, imshow, axis, savefig
from matplotlib.image import imread


# create a dataframe of positive and negative examples
# download cropped data folder from Onedrive
pos_folder = "../cropped_data/positive/raw"
neg_folder_1 = "../cropped_data/negative_combined/negative/mapped"
neg_folder_2 = "../cropped_data/negative_combined/negative/negative_unmapped"

total_positives = len(os.listdir(pos_folder))
total_negatives = len(os.listdir(neg_folder_1)) + len(os.listdir(neg_folder_2))

total_images = total_positives + total_negatives

counts_df = pd.DataFrame(
    {
        "Positive": [
            total_positives,
            str(np.round((total_positives / total_images) * 100)) + "%",
        ],
        "Negative": [
            total_negatives,
            str(np.round((total_negatives / total_images) * 100)) + "%",
        ],
    }
)

counts_df = counts_df.rename(index={0: "Count", 1: "Proportion"})
counts_df.to_csv("results/counts_df")

# get the recall and accuracies from the respective csv files in results folder

models_list = ["densenet", "inception", "vgg", "resnet"]
acc_models_list = []
recall_models_list = []

for model in models_list:
    df = pd.read_csv("results/" + model + "_test_accuracy.csv")
    acc_models_list.append(df["0"][0])

    df2 = pd.read_csv("results/" + model + "_test_recall.csv")
    recall_models_list.append(df2["0"][0])

test_summary_df = pd.DataFrame(
    {
        "model": ["DenseNet", "Inception", "VGG16", "ResNet"],
        "test_accuracy": acc_models_list,
        "test_recall": recall_models_list,
    }
)

test_summary_df.to_csv("results/test_summary_df", index=False)

# create bar chart for accuracy

model_acc_bar_chart = (
    alt.Chart(test_summary_df)
    .mark_bar(color="black")
    .encode(
        x=alt.X("test_accuracy", title="Test Accuracy"),
        y=alt.Y("model", title="Model", sort="x"),
    )
)


model_acc_bar_chart.save("image/model_acc_bar_chart.png")

# create bar chart for recall

model_recall_bar_chart = (
    alt.Chart(test_summary_df)
    .mark_bar(color="black")
    .encode(
        x=alt.X("test_recall", title="Test Recall"),
        y=alt.Y("model", title="Model", sort="x"),
    )
)

model_recall_bar_chart.save("image/model_recall_bar_chart.png")

# get file sizes
# get trained_models folder from OneDrive
file_paths = [
    "../trained_models/trained_models_May25/resnet.pt",
    "../trained_models/trained_models_May25/inception_bo_simple.pth",
    "../trained_models/trained_models_June2/densenet_final.pth",
    "../trained_models/trained_models_June2/vgg16-final.pth",
]

size_df = pd.DataFrame(
    {
        "Model": ["ResNet", "Inception", "DenseNet", "VGG16"],
        "Size (MB)": [
            np.round(os.path.getsize(file) / 1000000, 2) for file in file_paths
        ],  # gets size and converts bytes to MB
    }
)
size_df = size_df.sort_values("Size (MB)")
size_df.to_csv("results/models_size_comparison", index=False)  # saves df to results
