model_parameter = {
    "XGBRegressor": {
        'learning_rate': [.1, .01, .05, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "AdaBoost Regressor": {
        'learning_rate': [.1, .01, 0.5, .001],
        'loss':['linear','square','exponential'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }, 

}
