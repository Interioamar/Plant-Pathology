## **Plant Pathology (** Identify the category of foliar diseases in apple trees **)**

![apple_tree_image](https://user-images.githubusercontent.com/107593984/189394666-6402deb5-4ce1-4a61-be0f-793a617eb422.jpg)

Source:
<https://www.agrifarming.in/apple-tree-pests-and-diseases-control-management>

**Introduction:**

1.  Over the growing season, apple tress are under constant threat from
    a large number of insects, fungal, bacterial and viral pathogens,
    particularly in the Northeastern U.S.

2.  Based on the incidence and severity of infection by diseases and
    insects, impacts range from appearance, low marketability and poor
    quality of fruit, to decreased yield or complete loss of fruit or
    trees, causing huge economic losses. So timely deployment of disease
    management procedures are depends on early disease detection.
    Currently, disease diagnosis is done manually and is time-consuming
    and expensive

3.  'Plant Pathology Challenge' was part of the Fine-Grained Visual
    Categorization (FGVC) workshop at CVPR 2020. **Goal is to
    automatically classify disease symptoms with high accuracy without
    an expert plant pathologist**

<https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7>

**Terminologies:**

**Pathology:** Pathology is the study of the causes and effects of
disease or injury.

**Foliar disease:** It is a disease that impacts the leaves of a tree,
shrub, or other plants. And it is usually a response to an irritating
agent. The majority of the time, this is a fungal or fungal-type
organism.

**Problem Statement:**

1.  Categorization of different diseases based on leaf image.
    So here we need to classify a given image from apple leaf dataset into
    different disease and healthy leaf category. Also need to distinguish
    between multiple diseases in a single leaf.

2.  Currently this problem has a DL solution solved using off-the-shelf
    convolutional neural network (CNN). And there are great variance in
    symptoms due to age of infected tissues, genetic variations, and
    light conditions within trees decreases the accuracy of detection.
    So here we need to improve the accuracy of model for better classificatrion of foliar disease.

**Business Constraints**

1.  Incorrect disease prediction will adversly impact the type of
    procedure will be applied on the tree to cure the disease. Also
    incorrectly classifying healthy leaf to disease makes unnecessary
    increase in cost.

2.  Model can take a few seconds to evaluate the result. As here more
    importance is given to correct prediction than the prediction
    execution time

**About Dataset:**

-   This dataset has 3642 images and has labeled data for train images
    and is a Supervised learning(labelled) dataset

-   Given a photo of an apple leaf,we have to distinguish between leaves
    which are healthy, those which are infected with apple rust, those
    that have apple scab, and those with more than one disease.

General information about the dataset:

  1.  Total number of images =3642
  2.  Target columns : Healthy, Multiple diseases, Rust, Scab
  3.  Dataset is having both train.csv & test.csv files
  4.  train.csv is having total 5 columns i.e. image_id and 4 categories of target columns labelled in binary format.
  5.  Total number of datapoints in train.csv file are : 1821
  6.  Total number of datapoints in test.csv file are : 1821
  7.  Image_id: the unique key for datapoint
  
#### Train dataset classification overview 

![image2](https://user-images.githubusercontent.com/107593984/189395205-a76f587e-2580-4081-8bb2-cbadbb4f9410.png)


 This is a Multiclass Classification problem. The metric for evaluation in Kaggle is **ROC AUC of each predicted column.**
 
 Other Metrics: Confusion Matrix,F1 score can be used as the dataset is
 not balanced.

**TLDR:**

![TLDR](https://user-images.githubusercontent.com/107593984/189395043-5d96b04c-9bf1-4028-bb60-4179c04ae7f5.jpg)

Source: <https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.11390>

#### **Deployed model in heroku (** Identify the category of foliar diseases in apple trees **)**
https://plant-pathology.herokuapp.com/


