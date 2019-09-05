## Housing Feature Analysis
Connor Anderson   
  
**Blog Post URL: Written by Andy Luc** https://dev.to/rokaandy/logarithmic-transformation-in-linear-regression-models-why-when-3a7c  
**Slide Deck Presentation Link:** https://docs.google.com/presentation/d/1HIq4f5RjB3SX3YhBeCwdoo9FX8kIVZmTKjEwPk2GQfA/edit?usp=sharing

## Business Understanding:
We work for a real estate agency. Our agents have a group of customers who are looking to sell their homes in the near future, but they would like to know what renovations to add to their homes in order to increase the value and get the most return on investment. Our 3 main questions:

  1. What factors influence price the most?
  2. What factors are feasible for renovation
  3. How accurately can we predict the value these features will have on price?

## Data Understanding:
The dataset used for this model is the King County Housing Dataset. The dataset ranges from 2014-2015, with over 25,000 rows and 20 columns. The features included in this dataset is information about the house(sq. footage, waterfront, condition) and about the area surrounding the house(zipcode, neighbors lot size, etc.) The data was stored as a CSV file, and could easily be read in using Pandas. A heatmap and seveal other covariance charts were used in order to understand which variables were related and which variables may help the model perform best. 

## Data Prep
The dataset was orginally fairly clean, however null values in the dataset were present as well as type differences that needed to be fixed before modeling could be performed. The functions ```clean_housing_data``` and ```normalize_data``` were written in order to make the cleaning process smooth and easy. After intial EDA was performed, the columns with issues(waterfront, view, sqft_basement) were all addressed with these functions. The ```normalize_data function``` was created to help faclitate the normaliztion of several columns that were skewed by outliers. In order for a linear regression to be reliable and accurate the data should follow a normal distribution. To solve this, the log of each of the skewed columns was taken and created, helping normalize the data. Numpy was used at the end of the modeling process to bring the values back to human-readable terms.

## Modeling

The modeling process was very exploratory. We began but creating several very simple models in order to establish a baseline. These were simple linear regressions with one or two features, just to experiment and gauge what was effective. Our final model was a multi-linear regression model that used features were determined wer both effective, and which lacked colinearity. 

## Evaluation
The modeling process return an RMSE of around 135,000 dollars which can and will be improved upon - however this was a large leap forward from our baselines models. This appeared to be because we removed the outliers, which helped our metrics because they are sensitive to outliers. The model would surely improve if we limited the data down to the inter-quaritle section of the data which would help us predict most people's homes much more accuratly.
