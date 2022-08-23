# Weather Status Predictor From Images

## About the Project

predicting weather status has been a rich field to apply the technology of machine
learning (ML) and deep learning (DL) along with Convolutional Neural Network (CNN). There are so many examples of
projects with various level of accuracy and advance that predict or classify the weather status from given
satellite images in visible light, infrared ranges, or other light spectra. However, for this project I will do a
similar approach but with images taken with smart phones or digital cameras of a landscape views, then try to
predict the appearing weather status in that appearance. I used different algorithms for ML using only two classes, that is, *sunny* and *cloudy*. For CNN I used the entire dataset of more than 18K images with 5-class
weather status. Each image is 200x200 pixels with 3-channel colors.

## Data Dictionary

You can find the dataset in [Kaggle](https://www.kaggle.com/datasets/ammaralfaifi/5class-weather-status-image-classification), where more details about the data is given.

## Data Summary

| Class   |   Folder  |   Images Count  |
---       | ---       | ---             |
| Sunny   | sunny     | 6702             |
| Cloudy  | cloudy    | 6274             |
| Foggy   | foggy     | 1261             |
| Rainy   | rainy     | 1927             |
| Snowy   | snowy     | 1875             |
| Total   | -         | 18039            |
---
