# twitter_sentiment_analysis

The purpose of this task is to identify instances of hate speech in tweets. To simplify the definition, we consider a tweet to contain hate speech if it exhibits racist or sexist sentiments. Therefore, the objective is to distinguish tweets that are racist or sexist from those that are not.

In a formal sense, you are given a set of labeled tweets for training purposes. A label of '1' indicates that the tweet is classified as racist/sexist, while a label of '0' indicates that the tweet is not racist/sexist. Your goal is to predict the labels for a test dataset.

To facilitate model training, we provide a labeled dataset comprising 31,962 tweets. The dataset is presented as a CSV file, with each line containing a tweet ID, its corresponding label, and the tweet text.

data set link: https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/

## **Libraries Used** ##
* pandas
* matplotlib
* seaborn
* scikit-learn
* Transformers

## Algorithm used ##
* Logistic Regression
* A BERT model implementation is also done 


