# Sentiment Analysis on Amazon Alexa Reviews

## Introduction:
This project involves sentiment analysis on Amazon Alexa reviews using deep learning and traditional machine learning techniques. The objective is to classify the reviews as either positive or negative based on their content. The dataset consists of 36 lakh train data points used for training and validation, and 4 lakh test data points for evaluation.

## Project Overview
- Objective: Classify Amazon Alexa reviews as positive or negative.
- Dataset: 36 lakh train data points and 4 lakh test data points.
- Techniques: Recurrent Neural Network (RNN) with LSTM layers , Logistic Regression, Stochastic Gradient Descent and Naive Bayes.
- Frameworks: TensorFlow, Keras.
- Accuracy: Achieved 94.1% training accuracy and 94.2% testing accuracy using LSTM model.
- Deployment: Logistic Regression model is deployed using streamlit.

## Installation
Clone the repository:
```
git clone https://github.com/MPoojithavigneswari/Sentiment-Analysis.git
cd Sentiment-Analysis
```
Install the dependencies:
```
pip install -r requirements.txt
```

## Usage
Run the Streamlit app:
```
streamlit run streamlit_app.py
```
This generates a link and opens it in your web browser. Enter the any review to get its predicted sentiment.

## Dataset
The dataset isw large and consists of Amazon Alexa product reviews, which include the review text and the corresponding sentiment label (positive or negative). Click [here](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) to download the dataset.

## Deployment
The Streamlit app is deployed and hosted at [Sentiment Analysis Web App](https://sentiment-analysis-alexa.streamlit.app/).

## Results
The Logistic Rgrssion model achieved an accuracy of 87.5% on the test set. The Streamlit app provides a user-friendly interface for predicting the sentiment of new reviews.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
