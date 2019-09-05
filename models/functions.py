import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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
    
    print(f'Mean Absolute Error:', np.exp(metrics.mean_absolute_error(y_test, y_pred)))
    print(f'Mean Squared Error:', np.exp(metrics.mean_squared_error(y_test, y_pred)))
    print(f'Root Mean Squared Error:', np.exp(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
    predictions =  np.array(np.exp(lr.predict(X_test)))
    standardized_y= np.exp((y_test))
    print('RMSE(US DOLLARS)', np.sqrt(sum((standardized_y - predictions) ** 2) / len(standardized_y)))
    
    scores = cross_val_score(lr, X, y, cv=7)
    print('Cross Validated RMSE Scores', np.mean(scores))
    print('MAE')
    print(np.log(metrics.mean_absolute_error(standardized_y, predictions)))
    print('LR Coef')
    print(np.exp(lr.coef_))
    
    
def clean_housing_data(data):
    '''Cleans data by removing question marks, and redefines types.'''
    data['sqft_basement'] = data['sqft_basement'].replace("?",0)
    data['sqft_basement'] = data['sqft_basement'].astype(float)
    data['zipcode'] = data['zipcode'].astype(str)

    data = data[~np.isnan(data['waterfront'])]
    data = data[~np.isnan(data['view'])]

    final = data.drop(['id','yr_renovated','lat','long'], axis=1)
    return final

def normalize_data(data): 
    '''Normalize Housing Data '''
    data['log_sqft'] = np.log(data['sqft_living'])
    data['log_bedrooms'] = np.log(data['bedrooms'])
    data['log_yr_built'] = np.log(data['yr_built'])
    data['price'] = np.log(data['price'])
    return data 

def create_heatmap(data):
    '''Uses data to create a Correlation Matrix'''
    sns.set(style="white")

    # Create a covariance matrix
    corr = data.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11,9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220,10,as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr,mask=mask,cmap=cmap,vmax=1,center=0,square=True, 
                linewidth=.5, cbar_kws={'shrink': .5})

    ax.set_title('Multi-Collinearity of Features')