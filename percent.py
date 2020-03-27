import pandas as pd
data=pd.read_csv('coronapercent2.csv')
import pickle

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['State']=le.fit_transform(data['State'])
data['Hotspot_city']=le.fit_transform(data['Hotspot_city'])
data['travel_abroad']=le.fit_transform(data['travel_abroad'])
data['contact_abroad_people']=le.fit_transform(data['contact_abroad_people'])

X=data.iloc[:,:-1].values
y=data.iloc[:,4].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()
regressor.fit(X,y)

pickle.dump(regressor, open('percent.pkl','wb'))
