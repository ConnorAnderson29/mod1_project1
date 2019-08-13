def create_models(df, feature_cols):
    '''This is a function that outputs the Intercept, Coefficients, 3 types of Errors, and R-Squared'''
    X = df[feature_cols]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f'Intercept of the regression line:',lr.intercept_)
    print(f'Coefficients:',lr.coef_)
    print('\n')
    
    y_pred = lr.predict(X_test)

    
#     print(f'Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#     print(f'Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#     print(f'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#     print(f'R-Squared:',round(lr.score(X,y),3))
#     print('\n')
    
    print(f'Mean Absolute Error exp:', np.exp(metrics.mean_absolute_error(y_test, y_pred)))
    print(f'Mean Squared Error exp:', np.exp(metrics.mean_squared_error(y_test, y_pred)))
    print(f'Root Mean Squared Error:', np.exp(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
    predictions =  np.array(np.exp(lr.predict(X_test)))
    standardized_y= np.exp((y_test))
    print(np.sqrt(sum((standardized_y - predictions) ** 2) / len(standardized_y)))
    scores = cross_val_score(lr, X, y, cv=7)
    print('Cross Validated RMSE Scores', np.mean(scores))
    print('MAE')
    print(np.log(metrics.mean_absolute_error(standardized_y, predictions)))
    print('LR Coef')
    print(np.exp(lr.coef_))