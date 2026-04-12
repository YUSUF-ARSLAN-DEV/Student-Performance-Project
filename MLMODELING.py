import numpy as np 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score 
from dataengineering import load_data , preprocessing

# a method to evaluate all models 

def evaluate(model_name , y_true , y_pred ):
    rmse = np.sqrt(mean_squared_error(y_true,y_pred ))
    mae = mean_absolute_error(y_true,y_pred)
    r2 = r2_score(y_true , y_pred)
    print(f"\n{model_name }")
    print(f" RMSE : {rmse:.4f}")
    print(f" MAE : {mae:.4f}")
    print(f"r2 : {r2:.4f}")
    return {"model":model_name , "RMSE":rmse , "MAE": mae , "R2":r2 }

def train_baseline(X_train , y_train , X_val , y_val ):
    # baseline model which is lienar regression model 
    model = LinearRegression() 
    model.fit(X_train,y_train)
    predections = model.predict(X_val)
    results = evaluate("Linear Regression" , y_val , predections)
    return model , results 

if __name__ == "__main__":
    train , val , test = load_data() 
    X_train , X_val , scaler , X_test , y_train , y_val , y_test  = preprocessing(train,val,test)
    model , results = train_baseline(X_train , y_train , X_val , y_val )
    print(f"The Results For This model are as follows: {results}")