# NLP_Stock_Market_Sentiment_Analysis
# Market Sentiment Analysis Project

**Course:** NLP (Semester 6) - Pillai College of Engineering

## Project Overview
This project is part of the Natural Language Processing (NLP) course for Semester 6 students at Pillai College of Engineering. The project focuses on Market Sentiment Analysis, where we apply various Machine Learning (ML), Deep Learning (DL), and Language Models to classify financial texts such as news articles, tweets, financial reports, and statements into sentiment categories (positive, negative, neutral). This project involves exploring techniques like text preprocessing, feature extraction, model training, and evaluation to assess the effectiveness of different models in sentiment classification. You can learn more about the college by visiting the [official website of Pillai College of Engineering](https://www.pce.ac.in).

## Acknowledgements
We would like to express our sincere gratitude to the following individuals:

**Theory Faculty:**
- Dhiraj Amin
- Sharvari Govilkar

**Lab Faculty:**
- Dhiraj Amin
- Neha Ashok
- Shubhangi Chavan

Their guidance and support have been invaluable throughout this project.

## Project Title
**Market Sentiment Analysis using Natural Language Processing**

## Project Abstract
This project investigates the application of natural language processing techniques for market sentiment analysis using a synthetically generated dataset of tweets. The dataset consists of labelled statements with sentiment categories—positive, negative or neutral and the goal is to classify tweets into sentiment categories related to specific markets or stocks, aiding investors and analysts in predicting market trends and making informed decisions. The methodology includes data collection, data preprocessing (tokenization, stop word removal, stemming/lemmatization), and feature extraction using TF-IDF and word embeddings (e.g., Word2Vec, GloVe). For sentiment classification, traditional machine learning models like Naive Bayes and Support Vector Machine, as well as deep learning models such as RNN and CNN can be employed. The models are evaluated using metrics like accuracy, precision, recall, and F1-score to ensure reliability. This project demonstrates the potential of NLP in leveraging social media data for enhancing financial decision-making.

## Algorithms Used

### Machine Learning algorithms:
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest

### Deep Learning algorithms:
- Bidirectional Long Short-Term Memory (BiLSTM)
- Convolutional Neural Networks (CNN)
- LSTM (Long Short-Term Memory)

### Language Models:
- RoBERTa (Robustly Optimized BERT approach)
- BERT (Bidirectional Encoder Representations from Transformers)

## Comparative Analysis

| Model Type                                 | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| ------------------------------------------ | ------------ | ------------- | ---------- | ------------ |
| SVM (Support Vector Machine)               | 93.07        | 93.0          | 93.0       | 93.0         |
| Random Forest                              | 89.6         | 90.0          | 89.0       | 89.0         |
| Logistic Regression                        | 84.16        | 84.0          | 84.0       | 84.0         |
| Bidirectional Long Short-Term Memory (BiLSTM) | 93.6      | 93.7          | 93.6       | 93.6         |
| Convolutional Neural Networks (CNN)        | 93.1         | 93.1          | 93.1       | 93.0         |
| CNN-BiLSTM                                 | 93.1         | 93.3          | 93.1       | 93.1         |
| LSTM (Long Short-Term Memory)              | 91.1         | 91.2          | 91.1       | 91.1         |
| BERT (Bidirectional Encoder Representations from Transformers) | 94.5 | 94.6 | 94.5 | 94.5 |
| RoBERTa (Robustly Optimized BERT approach) | 95.5         | 95.6          | 95.5       | 95.5         |

## Conclusion
This Market Sentiment Analysis project highlights the effectiveness of various Machine Learning, Deep Learning, and Transformer-based models in classifying financial news, tweets, and reports into different sentiment categories. The comparative analysis demonstrates that RoBERTa, a transformer-based model, achieves the highest accuracy, precision, and recall, outperforming traditional ML models and deep learning architectures like CNN, LSTM, and BiLSTM. By evaluating different approaches, we gain insights into their strengths and limitations, allowing us to select the most optimal model for sentiment classification. The results indicate that while traditional models like SVM and Random Forest perform well, deep learning architectures like LSTMs and BiLSTMs enhance performance significantly. Ultimately, transformer-based models like BERT and RoBERTa provide superior results, making them the best choice for sentiment analysis in financial markets.
