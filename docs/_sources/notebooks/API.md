# API

After carefully evaluating the data and the model performances, we are now able to build a simple API to serve predictions.  
  
Despite the better performance of the XGBoost model across all metrics, we chose to build the API using the Logistic Regression model for the following reasons:  
  
- Performances are over-optimistic and the data-set constraints foster over-fitting. Logistic Regression is far less susceptible to over-fitting than XGBoost.  
  
- To keep things simple we want the API to apply a fixed threshold to all predictions. Logistic Regression makes this choice unambiguous by using a cost function.  
  
With the choice of the model out of the way, we built a simple API using FastAPI that can receive POST requests carrying all the features of a transactions in json format and send back the score given by the model, the current threshold and of course, the predicition.  
You can find some more details about the API in the README file for this project.  

