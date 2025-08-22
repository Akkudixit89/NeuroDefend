import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')
        tkns_ByDot = []
        for j in range(0, len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))
    if 'com' in total_Tokens:
        total_Tokens.remove('com')
    return total_Tokens
def train_model():
    # Load dataset and build model as before
    urls_data = pd.read_csv("url_features.csv")
    y = urls_data["malicious"]
    url_list = urls_data["URL"]
    
    vectorizer = TfidfVectorizer(tokenizer=makeTokens)
    X = vectorizer.fit_transform(url_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logit = LogisticRegression()
    logit.fit(X_train, y_train)

    accuracy = logit.score(X_test, y_test)
    return logit, vectorizer, accuracy


def get_test_predictions(model, vectorizer):
    # Sample URLs
    X_predict = [
        "google.com/search=jcharistech",
        "google.com/search=faizanahmad",
        "pakistanifacebookforever.com/getpassword.php/", 
        "www.radsport-voggel.de/wp-admin/includes/log.exe", 
        "ahrenhei.without-transfer.ru/nethost.exe ",
        "www.itidea.it/centroesteticosothys/img/_notes/gum.exe"
    ]

    # Vectorize and predict
    X_vectorized = vectorizer.transform(X_predict)
    predictions = model.predict(X_vectorized)

    return X_predict, predictions
