import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# Tokenizer for URLs
# -----------------------
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


# -----------------------
# Train multiple models
# -----------------------
def train_models():
    # Load dataset
    urls_data = pd.read_csv("url_features.csv")
    y = urls_data["malicious"]
    url_list = urls_data["URL"]
    
    vectorizer = TfidfVectorizer(tokenizer=makeTokens)
    X = vectorizer.fit_transform(url_list)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Logistic Regression
    logit = LogisticRegression()
    logit.fit(X_train, y_train)
    logit_acc = logit.score(X_test, y_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)

    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_acc = xgb_model.score(X_test, y_test)

    # Return all models + accuracies + vectorizer + full dataset
    models = {
        "Logistic Regression": logit,
        "Random Forest": rf,
        "XGBoost": xgb_model
    }
    accuracies = {
        "Logistic Regression": round(logit_acc*100, 2),
        "Random Forest": round(rf_acc*100, 2),
        "XGBoost": round(xgb_acc*100, 2)
    }
    return models, vectorizer, accuracies, urls_data


# -----------------------
# Get predictions on full dataset
# -----------------------
def get_all_predictions(models, vectorizer, urls_data):
    # Use *all* URLs from dataset
    X_predict = urls_data["URL"].tolist()
    
    # Vectorize
    X_vectorized = vectorizer.transform(X_predict)

    results = {}
    for model_name, model in models.items():
        preds = model.predict(X_vectorized)
        results[model_name] = [
            (url, "Malicious" if pred == 1 else "Benign") 
            for url, pred in zip(X_predict, preds)
        ]

    return results



# -----------------------
# Example run (for testing only)
# -----------------------
if __name__ == "__main__":
    models, vectorizer, accuracies, urls_data = train_models()
    results = get_test_predictions(models, vectorizer, urls_data)

    for model_name in models.keys():
        print(f"\n=== {model_name} ===")
        print(f"Accuracy: {accuracies[model_name]}%")
        for url, pred in results[model_name][:10]:  # show only first 10 for readability
            print(f"{url} --> {pred}")
