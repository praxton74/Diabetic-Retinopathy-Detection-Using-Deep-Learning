# Diabetic-Retinopathy-Detection-using-Deep-Learning

## Abstract
Diabetic Retinopathy(DR) is an eye complication caused due to diabetes and is one of the major causes of blindness among working population. Although this disease is fatal, timely treatment and control over the blood glucose level can help averse it to a great extent. However, due to varying complexities in the DR, its detection and diagnosis is difficult in the time-consuming manual diagnosis. Thus, we have worked on utilizing the power of Artificial Intelligence specifically Convolutional Neural Networks(CNNs) to automate the detection process using fundus images.
We used publicly available dataset on Kaggle and used EfficientNet models using pretrained ImageNet weights for better accuracy. We achieved a maximum accuracy of 87.66% on detection of DR on our model trained using Google Colab.

## Introduction
Diabetic Retinopathy is an eye condition in which diabetic patients lose their vision due to diabetic maculopathy and complications such as vitreous hemorrhage in which the blood vessels in the retina are affected leading to blindness[1].
In developed nations, it is the most frequent microvascular complication caused by diabetes and is the most common cause of blindness in the working-age population. It is estimated that by 2030,
there will be about 440 million diabetes patients in the age group 20-79 years worldwide in comparison to 285 million in 2010, accounting for an increase of 20% in diabetic patients in developed nations and 69% in developing nations[1]. Due to this expected rise in diabetic patients, the ophthalmic care of patients will also increase ,leading to a rise in the burden on the eye-care providers.
In most developing regions, detection of Diabetic Retinopathy is still a manual process that requires a well-trained clinician to examine the retina using photographs from a specialized low- power microscope attached to a camera. The clinician checks for the presence of lesions associated with the vascular abnormalities caused by the disease. This is a highly skilled human capital intensive process, and also can lead to unintended human errors. We look to solve this problem
through a deep learning model based on convolutional neural networks which will not only assist the care providers in diagnosis of diabetic retinopathy but also expedite the testing process[2].

## Datasets and Image Preprocessing

### Training Dataset
We used publicly available 2015 Diabetic Retinopathy dataset from Kaggle competition which contains 35,127 fundus images for training our model. The images are classified into 5 classes where label 0 indicates No Diabetic Retinopathy, label 1 indicates Mild, label 2 indicates Moderate, label 3 indicate Severe and label 4 indicates Proliferative Diabetic Retinopathy. There are varying number of images from each class and are in the below table:

| Label | Level of DR | Count of Images |
| --------------------- |:---:|:---:|
| 0 | No DR | 25810 |
| 1 | Mild | 5292 |
| 2 | Moderate | 2443 |
| 3 | Severe | 873 |
| 4 | Proliferative DR | 708 |

![alt text](https://github.com/sameer7483/Diabetic-Retinopathy-Detection-using-Deep-Learning/blob/main/image_distribution.png)

The training dataset contains images captured using cameras of different aspect ratios as a result some of eye images have higher black region which is not relevant for prediction. In addition to that, the shape of eye images are not uniform and are of varying shapes. Thus, we did Image processing to obtain a uniform dataset of eye images. We have also performed augmentation techniques to normalize the image using albumentations. The results are as below :

![alt text](https://github.com/sameer7483/Diabetic-Retinopathy-Detection-using-Deep-Learning/blob/main/original_fundus.png)

![alt text](https://github.com/sameer7483/Diabetic-Retinopathy-Detection-using-Deep-Learning/blob/main/transformed_fundus.png)

Link to the training dataset : https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data

### Test Dataset
We used publicly available 2019 Aptos Diabetic Retinopathy dataset from Kaggle competition which contains 3,663 fundus images for testing our model.
Link to the test dataset : https://www.kaggle.com/competitions/aptos2019-blindness-detection/


## Methodology
We have preprocessed the dataset to obtain uniform eye images and have used Convolutional Neural Network(CNN) model with EfficientNet architecture to train our model. Recently, CNNs have become highly popular for image classification tasks owing to their good performance and therefore we have utilized it for classification of Diabetic Retinopathy in different classes. Prior to
   
training our model using the training set, we utilized weights from a CNN trained on Imagenet for initializing our model based on EfficientNet Architecture to pretrain our model so that they have initialized weights before beginning the actual training. This process of pretraining is also known as Transfer Learning. We trained using EfficientNet architectures B2,B3,B4, and found that B3 performs the best among them for Diagnosis classification, where as B2 Performs the best for classification of level of DR.
We have used EfficientNet Architecture for our CNN model, which are family of scaled baseline networks designed on neural architecture search. These models achieve much better accuracy and efficiency than existing ConvNets and have been extensively used for transfer learning because of their state-of-the-art accuracy on some of the well-known datasets such as CIFAR-100(91.7%) and
Flowers(98.8%)[4].

## Results and Discussion
We have obtained two kinds of result:
a. Model predicts whether the image has DR or No DR(Binary Classification) b. Model predicts the class of the Diabetic Retinopathy i.e. 0, 1, 2, 3, 4.
Accuracy is defined as total number of correctly classified class over total number of images.
 
We have trained our model using EfficientNet-B2, EfficientNet-B3, EfficientNet-B4 and found out that the model trained on EfficientNet-B3 gave maximum accuracy of 87.66% on Binary Classification for identification of Diabetic Retinopathy whereas maximum accuracy for Multi- class classification was found when trained using EfficientNet-B2 which has an accuracy of 53.39%.

![alt text](https://github.com/sameer7483/Diabetic-Retinopathy-Detection-using-Deep-Learning/blob/main/bi-classification.png)

![alt text](https://github.com/sameer7483/Diabetic-Retinopathy-Detection-using-Deep-Learning/blob/main/class-classification.png)

## Conclusions
Diabetic Retinopathy is a fatal complication of diabetes and can lead to blindness. Early diagnosis of DR is important for an early treatment so as to prevent visual impairments. Consequently, an effective diagnostic method can help reduce the blindness caused due to DR. Therefore, We have worked on automating the diagnostic process using Convolutional Neural Network thus eliminating the time consuming manual-diagnostic method. We have obtained a high accuracy of 87.66% in identifying whether a fundus image have Diabetic Retinopathy or not which is quite significant and fast. In addition, we have obtained an accuracy of 53.39% in classifying the correct label to the fundus image. We believe that as a future work we can further improve the accuracy of these models by optimizing the hyperparameters.
