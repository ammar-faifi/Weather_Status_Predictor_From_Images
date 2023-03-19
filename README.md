# Weather Status Predictor From Images

## About the Project

The application of machine learning (ML) and deep learning (DL) technologies, accompanied by Convolutional Neural Network (CNN), 
has been extensively explored in the field of weather forecasting. Numerous projects have been executed with varying degrees of 
accuracy and sophistication, which use satellite images captured in visible light, infrared ranges, or other light spectra to 
predict or classify weather conditions. However, my project aims to apply a similar approach, but using landscape images taken 
with smartphones or digital cameras to predict the weather conditions depicted in the image. I utilized various ML algorithms 
with only two weather classes - sunny and cloudy - and CNN with the full dataset of over 18K images, with five weather classes. 
Each image is 200x200 pixels with 3-channel colors.

## Online Predictor

To predict an image [TRY NOW](https://web-production-df4e.up.railway.app/)


The final models that I reached are deployed on Railway
you can predict on binary classification or with 5-class classification.

![CNN Confusion Matrix](pages/CNN_confusion_matrix.png)

## Data

### About

I wanted to collect real fresh outdoors images with fire classes in a part of Misk Foundation Data Science Immersive project. With MS Bing API I collected and cleaned up to 1500 images for all classes. Further, I collected data from four kaggle datasets, their credits are below.

### Dictionary

You can find the dataset in [Kaggle](https://www.kaggle.com/datasets/ammaralfaifi/5class-weather-status-image-classification), where more details about the data is given.

### Data Summary

| Class   |   Folder  |   Images Count  |
---       | ---       | ---             |
| Sunny   | sunny     | 6702             |
| Cloudy  | cloudy    | 6274             |
| Foggy   | foggy     | 1261             |
| Rainy   | rainy     | 1927             |
| Snowy   | snowy     | 1875             |
| Total   | -         | 18039            |
---

### Sources

1 - Manually from Bing API
2 - https://www.kaggle.com/datasets/jagadeesh23/weather-classification
3 - https://www.kaggle.com/datasets/polavr/twoclass-weather-classification
4 - https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
5 - https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset
