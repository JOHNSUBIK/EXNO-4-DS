# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/da467f0d-673d-4668-8970-52d6b06bbd70)
```
df.head()
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/04672d0a-21d2-48c2-89d5-15b2b7a6e350)
```
df.dropna()
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/c477ef99-1466-4f40-bb32-d387c24bc06d)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/b41bc509-ac48-4054-854b-8aa0c2b621b3)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/d233f459-519a-4d5d-82e9-c452c6a5fff3)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/92117ba2-cdf8-443c-ab86-174ac0a35066)
```
from sklearn.preprocessing import Normalizer
scale=Normalizer()
df[['Height','Weight']]=scale.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/2f723c54-f0a0-47b5-a598-33e96cb95787)
```
from sklearn.preprocessing import MaxAbsScaler
scalen=MaxAbsScaler()
df[['Height','Weight']]=scalen.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/1b04052a-2de0-4dc2-b7de-964addbf2186)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/ce29b0df-9593-4989-8fbd-af9774b9c5f7)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/75ac6ad0-1456-4b2f-b22e-71e0f87c2588)

```
data.isnull().sum()
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/3b2db7aa-20a1-45d6-9d5e-c4474b3307f8)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/8a384716-7afa-4558-a9f0-d8312f5d5ee4)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/4d2c7021-3b92-4935-a836-2336b21983c1)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/2ced945a-839d-4d0a-b900-75283ea645d5)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/6a978115-8d4c-4fe5-a6af-c4d9ee2926ce)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/e4ae36da-8f18-4a91-be1b-f75502abdd71)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/2315e503-75b6-46d4-9260-22ccb2966e94)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/7e9f7472-e7d1-4b47-b4ff-e6b628c65d11)
```
x=new_data[features].values
x
```
![322205361-e72cd3ff-3f9a-432d-a40e-7596f464787b](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/92c181fb-e2af-42e3-8edb-b7b32698a74d)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![322205370-450c4d39-4e99-4437-be1f-b34ac64b1525](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/645e7e68-9319-4d28-8cf0-23b6bfe824cb)
```
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![322205391-14fcf028-4d87-460e-8757-18728e49ccee](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/5ed90b75-86f6-4a45-8b9d-ad71f3253d52)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![322205410-1b6440ef-c6c6-440f-9b86-317586560009](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/8e958d0a-5865-4e66-baa9-860e36d56098)
```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![322205416-f35864d0-6985-43b3-b821-1538039615a2](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/776fae62-58b1-44e3-8209-4e46647a2ae8)
```
data.shape
```
![322205424-a6544ea7-8884-4b26-a002-f1ada3f495e1](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/eb545f07-c123-432c-91a7-023f2509b463)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/19ad61f4-983e-4772-83f2-2ef81123090e)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![322205445-82f083f9-49c2-447d-a214-69dfa9ee6375](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/3be7c3d1-c251-421d-accc-c47745411d54)
```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![322205452-81020674-0b46-43f9-b4e7-aac8f7a95b46](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/ff20b72b-fbc6-4841-99e6-27bbc1ab45cf)
```
chi2,p, _, _ =chi2_contingency(contigency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![322205457-6409cb43-d8f0-4340-9a7f-3785a3cfe53b](https://github.com/JOHNSUBIK/EXNO-4-DS/assets/150279319/0fba2a99-d0d1-4fad-a866-b05b2545de52)

# RESULT:
Thus perform Feature Scaling and Feature Selection process and save the data to a file successfully.
