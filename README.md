# malware-detection
To keep the detection system up-to-date with evolving threats, the model can be retrained by collecting new malicious and legitimate URLs from trusted sources like PhishTank or Alexa. These new entries are combined with the existing dataset, and the model is retrained using the updated data. This helps the system learn new attack patterns.

Algorithm used: All of URLs in the dataset are labeled. We use 5-fold method to train-test our systems. After selecting features, we used four machine learning algorithms. They are

Linear Regression
Logistic Regression
Random Forest
Gaussian Naïve-Bayes

RESULTS: 
ALGORITHM -ACCURACY 
Linear Regression:- 93.04 
Logistic Regression:- 96.17 
Random Forest:- 82.20 
Naïve bayes:- 96.00
