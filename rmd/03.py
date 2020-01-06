
x=10.12


n=10

n

print("n=",n)

x=10.234

print(x)

print("x=%10.5f"%x)


a=True;a

b=False;b


10>3

10<3


print(3)


s='IlovePython';s

s[7]

s[2:6]

s+s

s*2


float('nan')


list1=[];list1

list1=['Python',786,2.23,'R',70.2]

list1

list1[0]

list1[1:3]

list1[2:]

list1*2

list1+list1[2:4]


X=[1,3,6,4,9];X

sex=[' 女',' 男',' 男',' 女',' 男']

sex

weight=[67,66,83,68,70];

weight


{}

dict1={'name':'john','code':6734,'dept':'sales'};dict1

dict1['code']

dict1.keys()

dict1.values()


dict2={'sex': sex,'weight':weight}; dict2


import numpy as np

np.array([1,2,3,4,5])

np.array([1,2,3,np.nan,5])


np.array(X)

np.arange(9)

np.arange(1,9,0.5)

np.linspace(1,9,5)


np.random.randint(1,9)

np.random.rand(10)

np.random.randn(10)


np.array([[1,2],[3,4],[5,6]])

A=np.arange(9).reshape((3,3));A


A.shape

np.empty([3,3])

np.zeros((3,3))

np.ones((3,3))

np.eye(3)


import pandas as pd


pd.Series()


X=[1,3,6,4,9]

S1=pd.Series(X);S1

S2=pd.Series(weight);S2

S3=pd.Series(sex);S3


pd.concat([S2,S3],axis=0)

pd.concat([S2,S3],axis=1)


S1[2]

S3[1:4]


pd.DataFrame()


pd.DataFrame(X)

pd.DataFrame(X,columns=['X'],index=range(5))

pd.DataFrame(weight,columns=['weight'],index=['A','B','C','D','E'])


df1=pd.DataFrame({'S1':S1,'S2':S2,'S3':S3});df1

df2=pd.DataFrame({'sex':sex,'weight':weight},index=X);df2


df2['weight2']=df2.weight**2; df2


del df2['weight2']; df2


df3=pd.DataFrame({'S2':S2,'S3':S3},index=S1);df3

df3.isnull()

df3.isnull().sum()

df3.dropna()

#df3.dropna(how = 'all')


df3.sort_index()

df3.sort_values(by='S3')


BSdata=pd.read_csv("../data/BSdata.csv",encoding='utf-8')

BSdata[6:9]


BSdata=pd.read_excel('../data/DaPy_data.xlsx','BSdata');BSdata[-5:]


BSdata=pd.read_clipboard();

BSdata[:5]


BSdata.to_csv('BSdata1.csv')


BSdata.info()

BSdata.head()

BSdata.tail()


BSdata.columns


BSdata.index


BSdata.shape

BSdata.shape[0]   # 行数

BSdata.shape[1]   # 列数


BSdata.values


BSdata.身高 #取一列数据，BSdata['身高']


BSdata[['身高','体重']]

BSdata.iloc[:,2]

BSdata.iloc[:,2:4]


BSdata.loc[3]

BSdata.loc[3:5]


BSdata.loc[:3,['身高','体重']]

BSdata.iloc[:3,:5]


BSdata[BSdata['身高']>180]

BSdata[(BSdata['身高']>180)&(BSdata['体重']<80)]


BSdata['体重指数']=BSdata['体重']/(BSdata['身高']/100)**2

round(BSdata[:5],2)


BSdata.iloc[:3,:5].T


pd.concat([BSdata.身高, BSdata.体重],axis=0)

pd.concat([BSdata.身高, BSdata.体重],axis=1)


for i in range(1,5):

    print(i)


fruits = ['banana', 'apple',  'mango']

for fruit in fruits:

   print('当前水果 :', fruit)


for var in BSdata.columns:

    print(var)


a = -100

if a < 100:

    print("数值小于100")

else:

    print("数值大于100")


-a if a<0 else a



def 函数名（参数1，参数2，…）：

      函数体

      return 语句


import numpy as np

x=[1,3,6,4,9,7,5,8,2];x


def xbar1(x):

   n=len(x)

   xm=sum(x)/n

   xm


def xbar2(x):

   n=len(x)

   xm=sum(x)/n

   return(xm)


xbar1(x)

xbar2(x)


np.mean(x)


def SS1(x):

    n=len(x)

    ss=sum(x**2)-sum(x)**2/n

    return(ss)


SS1(S1)


def SS2(x):

    n=len(x)

    xm=sum(x)/n

    ss=sum(x**2)-sum(x)**2/n

    return[x**2,n,xm,ss]


SS2(S1)


SS2(S1)[0]

SS2(S1)[1]

SS2(S1)[2]

SS2(S1)[3]


type(SS2(S1))

type(SS2(S1)[3])

