# Credit Card Fraud Detection: From Exploratory Data Analysis to Inference API  
  
This project is an end-to-end exploration of credit card fraud detection under realistic constraints: extreme class imbalance, temporal dependencies, limited labeled data, and operational decision-making requirements.  
  
The goal is not leaderboard optimization, but to explore how model choice, calibration, thresholding, and ranking strategies interact in practice, and to deliver a deployable prediction service.  
  
It builds on a transaction database collected during a research collaboration between Worldline and ULB (Université Libre de Bruxelles). You can find more details about the dataset here:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
  
Project structure
├── walkthrough/ <--Notebooks for analysis and model training
├── api/ <--Prediction API + Docker setup
└── README.md  
  
## Walkthrough overview  
  
The walkthrough notebooks document a step-by-step approach to fraud detection on a highly imbalanced dataset (≈0.17% frauds, 48 hours of data).  
After some exploratory data analysis, I trained a Logistic Regression model and introduced the main metrics used for high-imbalance classification problems (ROC and PR). I then trained a Gradient Boosted Tree, a staple of fraud detection problems, to compare and contrast performance and inference strategies with Logistic Regression.  
  
Below is a quick summary of those notebooks:  

**Key challenges:**  
  
- Severe class imbalance  
- Strong temporal effects and leakage risks  
- Limited positive samples for calibration  
- Asymmetric error costs  
- Dataset shortcomings  
- Limited time span (2 days of data), which leads to overly optimistic models as patterns are easier to identify over a short window  
- PCA-transformed features aimed at preserving customer privacy, which prevents feature engineering  
  
**Main finding:s**  
  
- Time-aware splitting is critical to avoid inflated performance  
- Calibration can fail when positive samples are scarce  
- High imbalance makes models overconfident and scores diverge from true probability estimates  
- Threshold-based optimization using a cost function works well for Logistic Regression  
- Ranking-based metrics (Recall@K, Precision@K) are better suited for tree-based models  
  
**Final model choice:**  
  
Despite XGBoost achieving slightly better ranking performance, I chose the Logistic Regression model calibrated using Platt scaling for the following reasons:  
  
- More reliable probability estimates  
- Stable threshold behavior  
- Predictable operational performance  
- In a realistic production setting, the better ranking performance and interpretability of Gradient Boosted Trees would likely - - outweigh the benefits of Logistic Regression’s more stable threshold selection.  
  
## Prediction API  
  
The project includes a production-style API exposing the fraud model through a /predict endpoint built with FastAPI.  
  
**How to run the API:**
  
- With Docker:  
    After downloading the project, from the api/ folder:  
    - Build the image: docker build -t fraud-api .  
    - Run the container: docker run -p 8000:8000 fraud-api  
  
- Directly (without Docker):  
    After downloading the project, from the api/ folder:  
    - Install the dependencies listed in requirements-api.txt  
    - Run: uvicorn app:app --reload  
  
In both cases, prediction requests will be served at: http://localhost:8000/predict  
  
To send the feature values required for prediction, you can:  
  
- Use the built-in FastAPI documentation: http://localhost:8000/docs  
  
- Send a POST request using curl:  
    curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
    "Time": 0,
    "feature_1": 0,
    "feature_2": 0,
    "feature_3": 0,
    "feature_4": 0,
    "feature_5": 0,
    "feature_6": 0,
    "feature_7": 0,
    "feature_8": 0,
    "feature_9": 0,
    "feature_10": 0,
    "feature_11": 0,
    "feature_12": 0,
    "feature_13": 0,
    "feature_14": 0,
    "feature_15": 0,
    "feature_16": 0,
    "feature_17": 0,
    "feature_18": 0,
    "feature_19": 0,
    "feature_20": 0,
    "feature_21": 0,
    "feature_22": 0,
    "feature_23": 0,
    "feature_24": 0,
    "feature_25": 0,
    "feature_26": 0,
    "feature_27": 0,
    "feature_28": 0,
    "Amount": 0
    }'  
  
You can also check the API status at: http://localhost:8000/health  
  
**Additional implementation notes:**  
  
- The API exposes a /version endpoint to surface key library versions, improving traceability and reproducibility across environments  
- Basic API tests are included to validate request/response behavior and ensure the service runs correctly after changes  
- Docker is used to guarantee environment consistency and simplify deployment across systems  
  
## Outlook  
  
This project was a great learning experience, and I had a lot of fun implementing and exploring the metrics associated to high-imbalance classification problems. My biggest takeaway is that in machine learning, the most important step is choosing the right tool for the job.  
In the future, I would like to revisit this project using a different dataset — ideally one where feature engineering is possible and where a longer time horizon allows for more realistic time-series cross-validation and model evaluation. 
  
Many thanks to the ULB research team for making this data publicly available.