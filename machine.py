
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

df=fetch_california_housing()
x=df.data
y=df.target
xt,xte,yt,yte=train_test_split(x,y,test_size=0.3)
from sklearn.tree import DecisionTreeRegressor
r=DecisionTreeRegressor()
r.fit(xt,yt)
yp=r.predict(xte)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(yte,yp)
r2s=r2_score(yte,yp)
for i in range(10):
  print(yte[i],yp[i])
import matplotlib.pyplot as p
x=df.feature_names
y=r.feature_importances_
p.figure(figsize=(12,8))
p.bar(x,y,color="red")
p.xlabel("featurers")
p.ylabel("imp")
p.title("xx")
p.show()
print("mean squared error:",mse)
print("R2score:",r2s)
