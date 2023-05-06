from model import HousePricingModel
import pandas as pd


data = pd.read_csv('data/housing.csv')
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

model = HousePricingModel(
    n_estimators=100,
    max_depth=5)

model.fit(X,y)

model.save('model/model_1')

predict = model.predict(X)
print(predict)
