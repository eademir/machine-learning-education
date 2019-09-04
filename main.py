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
from scipy.stats import norm, shapiro, levene, f_oneway, bartlett
import seaborn as sns
from scipy import stats
import pylab as pl
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf


ax = plt.gca()

iris=pd.read_csv("iris.csv")
iris.head()

ax.scatter(iris['sepal_length'],iris['sepal_width'])#verimi cizdirdim ve asagidaki kodlarla baslik ekledim kolonlara

colors={'setosa':'r','versicolor':'g','virginica':'b'}#her bir cicek türüne renk atadım

fig,ax=plt.subplots()
for i in range (len(iris['sepal_length'])):
    ax.scatter(iris['sepal_length'][i],iris['sepal_width'][i],
               color=colors[iris['species'][i]])
    
ax.set_title('Iris_Dataset')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
#tanımladıgımız renk koduna gidip her bir cicegin sahip oldugu tipe göre renk atadık 
#datamı tek tek gez datam setosaysa kırmızı yap vs dedik 
#suana akdar sadece yaprak uzunluguna ve genisligine baktık 
    
    
#petal bilgisiyle cizdirdigmizde;    
fig,ax=plt.subplots()
for i in range (len(iris['sepal_length'])):
    ax.scatter(iris['petal_length'][i],iris['petal_width'][i],
               color=colors[iris['species'][i]])
    
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

survived_sex = pd.crosstab(index = titanic["Survived"], 
                           columns = titanic["Sex"])
survived_sex
survived_sex.index = ["died","survived"]
survived_sex.head()

survived_class = pd.crosstab(index = titanic["Survived"], 
                             columns = titanic["Pclass"])
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
survived_class = pd.crosstab(index = titanic["Survived"], 
                             columns = titanic["Pclass"], margins = True)
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

# =============================================================================
# popülasyon normal dağılımdan geliyorsa, eminsek, z-testi
# popülasyon normal dağılımdan gelmiyorsa t-test
# =============================================================================

veri_normal = norm.rvs(size = 10000, loc = 0, scale = 1)

ax = sns.distplot(veri_normal)
ax = sns.distplot(veri_normal, bins = 100, kde = False, color = "orange", 
                  hist_kws = {"linewidth": 15, "alpha":1})

ax.set(xlabel = "Normal", ylabel = "Frekans")

#t-testi

olcumler = np.array([17, 160, 234, 149, 145, 107, 197, 75, 201, 225, 211, 119, 
               157, 145, 127, 244, 163, 114, 145,  65, 112, 185, 202, 146,
               203, 224, 203, 114, 188, 156, 187, 154, 177, 95, 165, 50, 110, 
       216, 138, 151, 166, 135, 155, 84, 251, 173, 131, 207, 121, 120])
sns.distplot(olcumler)
stats.describe(olcumler)
pd.DataFrame(olcumler).plot.hist()

stats.probplot(olcumler, dist = "norm", plot = pl, )
pl.show()

sm.qqplot(olcumler, line = "s")
pl.show()

#tests shopiro-wilks

stat, p = shapiro(olcumler)

print("Statistics = %.3f, p = %.3f" % (stat, p))

#2. gün

alpha = 0.05
if p < alpha:
    print("Örneklem Normal Dağılımdan gelmektedir") 
    #z-testi
else:
    print("Örneklem Normal Dağılımdan Gelmemektedir") 
    #t-tesi

stats.ttest_1samp(olcumler, popmean = 170)

stats.t.ppf(q = 0.025, df = len(olcumler) -1) #0,025lik alana denk gelen nokta
stats.t.cdf(x = -2.1753117985877966, df = len(olcumler))*2

sms.DescrStatsW(olcumler).tconfint_mean()

A = pd.DataFrame([30,27,21,27,29,30,20,20,27,32,35,22,24,23,25,27,23,27,23,25,
                  21,18,24,26,33,26,27,28,19,25])
  
B = pd.DataFrame([37,39,31,31,34,38,30,36,29,28,38,28,37,37,30,32,31,31,27,32,
                  33,33,33,31,32,33,26,32,33,29])

A_B = pd.concat([A,B],axis = 1)
A_B
A_B.columns = ["A", "B"]
A_B.head()

GRUP_A = np.arange(len(A))
GRUP_A = pd.DataFrame(GRUP_A)
a = pd.concat([A, GRUP_A], axis = 1)
A.head()

AB = pd.concat([A, B], axis = 1)
AB
AB.columns = ["GELIR", "GRUP"]
AB.head()

sns.boxplot(x = "GRUP", y = "GELIR", data = AB)

shapiro(A_B.A)
shapiro(A_B.B)

#levene test

levene(A_B.A, A_B.B)

stats.ttest_ind(A_B["A"], A_B["B"], equal_var = True)

oncesi = pd.DataFrame([123,119,119,116,123,123,121,120,117,118,121,121,123,119,
                       121,118,124,121,125,115,115,119,118,121,117,117,120,120,
                       121,117,118,117,123,118,124,121,115,118,125,115])
sonrasi = pd.DataFrame([118,127,122,132,129,123,129,132,128,130,128,138,140,130
                        ,134,134,124,140,134,129,129,138,134,124,122,126,133,127
                        ,130,130,130,132,117,130,125,129,133,120,127,123])

BIRLIKTE=pd.concat([oncesi,sonrasi],axis=1) #axis 1 olunca kolon olarak ekler 
BIRLIKTE.columns=["Öncesi","Sonrasi"]
BIRLIKTE.head()

GRUP_ONCESI=np.arange(len(oncesi))
GRUP_ONCESI=pd.DataFrame(GRUP_ONCESI)
GRUP_ONCESI[:]="ONCESI"
A=pd.concat([oncesi,GRUP_ONCESI],axis=1)

GRUP_SONRASI=np.arange(len(oncesi))
GRUP_SONRASI=pd.DataFrame(GRUP_SONRASI)
GRUP_SONRASI[:]="SONRASI"
B=pd.concat([sonrasi,GRUP_SONRASI],axis=1)
#her bir gruba "öncesi" ve  "sonrasi" şeklinde isim verdik.
AB=pd.concat([A,B])#verilerimizi alt alta birleştirdik
AB.columns=["SKOR","ONCESI_SONRASI"]#limdeki skorların öncesine mi sonrasinami ait oldugunu cikariacam 
AB.head()
sns.boxplot(x="ONCESI_SONRASI",y="SKOR",data=AB)
#şimdi varsayımlarımızı kontrol edicez.shapirov ile 
shapiro(oncesi)  #(0.9543656706809998, 0.10722451657056808)  sonucu cıktı p valum(0.1) alfadan(0.05) büyük oldugu icin h0 ı kabul ediyorum
shapiro(sonrasi) # (0.9780089259147644, 0.6159515380859375) sonucuu cıktı ve p valuem(0.6) alfadan (0.05) den büüyk oldugu  icin h0 ı kabul ettik .

stat, p = shapiro(BIRLIKTE)

alpha = 0.025
if p < alpha:
    print("Örneklem normal dağılımdan gelmektedir")
else:
    print("Örneklem normal dağılıma sahip değil")

# =============================================================================
# eksik alan
# =============================================================================

A = pd.DataFrame([28,33,30,29,28,29,27,31,30,32,28,33,25,29,27,31,31,30,31,34,
                  30,32,31,34,28,32,31,28,33,29])
B = pd.DataFrame([31,32,30,30,33,32,34,27,36,30,31,30,38,29,30,34,34,31,35,35,
                  33,30,28,29,26,37,31,28,34,33])
C = pd.DataFrame([40,33,38,41,42,43,38,35,39,39,36,34,35,40,38,36,39,36,33,35
                  ,38,35,40,40,39,38,38,43,40,42])

#eksik

ABC = pd.concat([A, B, C], axis = 1)
ABC.columns = ["GRUP_A", "GRUP_B", "GRUP_C"]
ABC.describe()

stat, p = shapiro(ABC["GRUP_A"])
print("Statistics = %.3f, p_value = %.3f" % (stat, p))

stat, p = shapiro(ABC["GRUP_B"])
print("Statistics = %.3f, p_value = %.3f" % (stat, p))

stat, p = shapiro(ABC["GRUP_C"])
print("Statistics = %.3f, p_value = %.3f" % (stat, p))

levene(ABC["GRUP_A"], ABC["GRUP_A"], ABC["GRUP_C"])

f_oneway(ABC["GRUP_A"], ABC["GRUP_B"], ABC["GRUP_C"])

iris = sns.load_dataset("iris.csv")

iris.boxplot(column = "sepal_length", by = "species", figsize = (10, 10))
iris.boxplot()

f, axs = plt.subplots(2,2,figsize=(16,16))

ax = f.add_subplot(221)
plt.hist(iris['sepal_length'],bins=20, color='green')
plt.title("Sepal length in cm")
plt.xlabel("sepal length cm")
plt.ylabel("Count")

ax2 = f.add_subplot(222)
plt.hist(iris['sepal_width'],bins=20, color='blue')
plt.title("Sepal width in cm")
plt.xlabel("sepal width cm")
plt.ylabel("Count")
ax2 = f.add_subplot(223)
plt.hist(iris['petal_length'],bins=20, color='red')
plt.title("Petal length in cm")
plt.xlabel("Petal length cm")
plt.ylabel("Count")


ax2 = f.add_subplot(224)
plt.hist(iris['petal_width'],bins=20, color='yellow')
plt.title("Petal width in cm")
plt.xlabel("Petal width cm")
plt.ylabel("Count")

f=plt.figure(1)
iris.boxplot(column="sepal_length", by="species", figsize=(8,8))
f.show()
g=plt.figure(2)
iris.boxplot(column="sepal_width", by="species", figsize=(8,8))
g.show()

iris.plot.hist(subplots=True, layout=(2,2), figsize=(20, 20), bins=20)

grps = pd.unique(iris.species.values)
grps

for name in grps:
    print(name, shapiro(iris['sepal_length'][iris['species'] == name]))


#eğer data normal dağılımdan geliyorsa bartlett kullanmak daha güçlü gelmiyorsa levene
bartlett(iris['sepal_width'][iris['species']  == 'setosa'], 
               iris['sepal_width'][iris['species']  == 'versicolor'], 
               iris['sepal_width'][iris['species']  == 'virginica'])

levene(iris['sepal_width'][iris['species']  == 'setosa'], 
               iris['sepal_width'][iris['species']  == 'versicolor'], 
               iris['sepal_width'][iris['species']  == 'virginica'])

bartlett(iris['petal_width'][iris['species']  == 'setosa'], 
               iris['petal_width'][iris['species']  == 'versicolor'], 
               iris['petal_width'][iris['species']  == 'virginica'])

levene(iris['sepal_width'][iris['species']  == 'setosa'], 
               iris['petal_width'][iris['species']  == 'versicolor'], 
               iris['petal_width'][iris['species']  == 'virginica'])


bartlett(iris['petal_length'][iris['species']  == 'setosa'], 
               iris['petal_length'][iris['species']  == 'versicolor'], 
               iris['petal_length'][iris['species']  == 'virginica'])

levene(iris['sepal_length'][iris['species']  == 'setosa'], 
               iris['petal_length'][iris['species']  == 'versicolor'], 
               iris['petal_length'][iris['species']  == 'virginica'])

f_oneway(iris['sepal_length'][iris['species']  == 'setosa'], 
               iris['petal_length'][iris['species']  == 'versicolor'], 
               iris['petal_length'][iris['species']  == 'virginica'])

f_oneway(iris['sepal_width'][iris['species']  == 'setosa'], 
               iris['petal_width'][iris['species']  == 'versicolor'], 
               iris['petal_width'][iris['species']  == 'virginica'])

results = smf.ols("sepal_length ~ (species)", data = iris).fit()
results.summary()



















































