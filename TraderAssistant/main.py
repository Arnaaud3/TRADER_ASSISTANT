from src.stock_market_data import historical_market_data as hmd
from src.stock_market_data import preprocess_market_data as pmd
from src.analysis_market_data import ia_prediction as ia
from src.analysis_market_data import pattern_detection as pad

def main():
    data = hmd.Historical_market_data("BTC-USD","1d").marketData
    preprocess_data = pmd.Preprocess_market_data(data)
    data_per_month = preprocess_data.create_subset_dataframes()
    for month,data in data_per_month.items():
        try:
            pattern_detection = pad.Pattern_detection(data)
            pattern_detection.plot_trendline()
        except:
            continue
    idx_train,idx_test = preprocess_data.create_train_test_indexes()
    X_train,X_test,y_train,y_test = preprocess_data.create_train_test_dataset(idx_train,idx_test)
    moving_average = preprocess_data.moving_average
    dp = preprocess_data.dp
    ia_pred = ia.IA_Prediction(X_train,X_test,y_train,y_test,moving_average,dp) 
    linear_reg_model = ia_pred.linear_regression_predict_trend()
    print("Prediction from a linear regression : \n")
    y_prediction = {"LinearRegressor_train_dataset":ia_pred.linear_regression_prediction_train(linear_reg_model),
                    "LinearRegressor_test_dataset":ia_pred.linear_regression_prediction_test(linear_reg_model)}
    # print(y_prediction)
    # ia_pred.plot_moving_average_prediction_train_dataset(y_prediction["LinearRegressor_train_dataset"])
    # ia_pred.plot_moving_average_prediction_test_dataset(y_prediction["LinearRegressor_test_dataset"])
    print('\n')
    y_forecast = ia_pred.ia_trend_forecasting(linear_reg_model)
    ia_pred.plot_trend_forecasting(y_prediction["LinearRegressor_train_dataset"],y_prediction["LinearRegressor_test_dataset"],y_forecast)
    return None
if __name__ == "__main__":
    main()