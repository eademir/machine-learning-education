# =============================================================================
# import pandas as pd 
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# data = pd.read_csv('creditcard.csv')1
# data.info()
# 
# data.corr()
# 
# f,ax = plt.subplots(figsize=(10, 10))
# sns.heatmap(data.corr(), annot=True, linewidths=.100, fmt= '.1f',ax=ax)
# plt.show()
# 
# data.head()
# 
# data.V3.plot(kind = 'line', color = '#4253f4',label = 'V3',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
# data.V4.plot(color = '#424244',label = 'V4',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')
# plt.legend(loc='upper right')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.title('Line Plot')
# plt.show()
# 
# plt.plot(data)
# plt.show()
# 
# data.plot(kind='scatter', x='V3', y='V4',alpha = 0.5,color = 'b')
# plt.xlabel('V3')            
# plt.ylabel('V4')
# plt.title('Scatter Plot')  
# plt.show()
# 
# data.V3.plot(kind = 'hist',bins = 100,figsize = (12,12))
# plt.show()
# 
# data.tail()
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
ax = plt.gca()

iris=pd.read_csv("iris.csv")
iris.head()

ax.scatter(iris['sepal_length'],iris['sepal_width'])#verimi cizdirdim ve asagidaki kodlarla baslik ekledim kolonlara

colors={'setosa':'r','versicolor':'g','virginica':'b'}#her bir cicek türüne renk atadım

fig,ax=plt.subplots()
for i in range (len(iris['sepal_length'])):
    ax.scatter(iris['sepal_length'][i],iris['sepal_width'][i],color=colors[iris['species'][i]])
    
ax.set_title('Iris_Dataset')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
#tanımladıgımız renk koduna gidip her bir cicegin sahip oldugu tipe göre renk atadık 
#datamı tek tek gez datam setosaysa kırmızı yap vs dedik 
#suana akdar sadece yaprak uzunluguna ve genisligine baktık 
    
    
#petal bilgisiyle cizdirdigmizde;    
fig,ax=plt.subplots()
for i in range (len(iris['sepal_length'])):
    ax.scatter(iris['petal_length'][i],iris['petal_width'][i],color=colors[iris['species'][i]])
    
ax.set_title('Iris_Dataset')
ax.set_xlabel('petal_length')

columns = iris.columns.drop(['species'])
x_data = range(0, iris.shape[0])
fig, ax = plt.subplots()
for column in columns:
    ax.plot(x_data, iris[column])
ax.set_title('Iris Dataset')
#ax.legend() #renkleriin hangisine ait olduğunu göstermek için sol üst köşeye koyulan tabloydu ama çalışmadı

plt.figure(figsize = (5,5))
x = iris["sepal_length"]
plt.hist(x, bins = 100, color = "#0dafef")
plt.title("Sepal Length in CM")
plt.xlabel("SpalLengthCm")

#eğer datamızda karşı görüş varsa, aykırı gözlem varsa, ortalama yerine medyan bakılır

#IQR ???
# Xoutlier = {x | x < Q1 - 1.5IQR|| x > Q3 + 1.5IQR, IQR = Q3 - Q1}

plt.figure(figsize = (10,7))
iris.boxplot()

#correlation
# petal length and sepal have length direct proportion. 
corr = iris.corr()
corr

#renkli gösterim
plt.matshow(iris.corr())
plt.show

rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10,10))
corr = df.corr()
corr.style.background_gradient(cmap = "coolwarm")

#titanic train csv imported from https://github.com/caglarmert/UVBMOB

titanic = pd.read_csv("titanic_train.csv")
titanic.info()

survived_sex = pd.crosstab(index = titanic["Survived"], columns = titanic["Sex"])
survived_sex
survived_sex.index = ["died","survived"]
survived_sex.head()

survived_class = pd.crosstab(index = titanic["Survived"], columns = titanic["Pclass"])
survived_class.columns = ["class1", "class2", "class3"]
survived_class.head()

char_cabin = titanic["Cabin"].astype(str)
new_Cabin = np.array([cabin[0] for cabin in char_cabin])
titanic["Cabin"] = pd.Categorical(new_Cabin)
titanic.info()

my_tab = pd.crosstab(index = titanic["Survived"], columns = "count")
my_tab
pd.crosstab(index = titanic["Cabin"], columns = "count")

pd.crosstab(index = titanic["Cabin"], columns = titanic["Survived"])

# =============================================================================
# survived_class = pd.crosstab(index = titanic["Survived"], columns = titanic["Pclass"], margins = False)
# survived_class
# =============================================================================
survived_class = pd.crosstab(index = titanic["Survived"], columns = titanic["Pclass"], margins = True)
survived_class

survived_class.columns = ["class1", "class2", "class3", "rowtotal"]
survived_class.index = ["died", "survived", "coltotal"]
survived_class
survived_class/survived_class.ix["coltotal", "rowtotal"]

survived_class/len(titanic["Survived"])

d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}
print(d)

df = pd.DataFrame(d)
print(df)

print(df.count())
print(df.sum())
print(df.mean())
print(df.median())
print(df.mode())
print(df.std())
print(df.min())
print(df.max())
#print(df.abs())
#print(df.prod())
print(df.cumsum())
#print(df.cumprod())






















￼























