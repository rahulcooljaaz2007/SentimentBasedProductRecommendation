import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


class SentimentBaseProductRecommenderModel:

    ROOT_PATH = "pickle/"
    MODEL_NAME = "sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    USER_RATINGS = "user_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        self.model = pickle.load(open(
            SentimentBaseProductRecommenderModel.ROOT_PATH + SentimentBaseProductRecommenderModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(
            SentimentBaseProductRecommenderModel.ROOT_PATH + SentimentBaseProductRecommenderModel.VECTORIZER)
        self.user_rating = pickle.load(open(
            SentimentBaseProductRecommenderModel.ROOT_PATH + SentimentBaseProductRecommenderModel.USER_RATINGS, 'rb'))
        self.data = pd.read_csv("dataset/sample30.csv")
        self.cleaned_data = pickle.load(open(
            SentimentBaseProductRecommenderModel.ROOT_PATH + SentimentBaseProductRecommenderModel.CLEANED_DATA, 'rb'))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    # function to get the top product 20 recommendations for the user

    def get_recommended_product_by_user(self, user):        
        return list(self.user_rating.loc[user].sort_values(ascending=False)[0:20].index)

        
    def get_sentiment_based_recommendations(self,user):
      if (user in self.user_rating.index):
        # get the product recommedation using the trained ML model
        recommendations =  self.get_recommended_product_by_user(user)
        filtered_data = self.cleaned_data[self.cleaned_data.id.isin(recommendations)]        
        X =  self.vectorizer.transform(filtered_data["clean_reviews_text"].values.astype(str))
        filtered_data["predicted_sentiment"]= self.model.predict(X)
        temp = filtered_data[['name','predicted_sentiment']]
        temp_grouped = temp.groupby('name', as_index=False).count()
        temp_grouped["pos_review_count"] = temp_grouped.name.apply(lambda x: temp[(temp.name==x) & (temp.predicted_sentiment==1)]["predicted_sentiment"].count())
        temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
        temp_grouped['pos_sentiment_percent'] = np.round(temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100,2)
        sorted_products= temp_grouped.sort_values('pos_sentiment_percent', ascending=False)[0:5]
        return pd.merge(self.data, sorted_products, on="name")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
      else:
        print(f"User name {user} doesn't exist")
        return None
