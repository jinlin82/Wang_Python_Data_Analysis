import math

import numpy as np

import matplotlib.pyplot as plt

x=np.linspace(0,2*math.pi);x

#fig,ax=plt.subplots(2,2,figsize=(15,12))

plt.plot(x,np.sin(x))

plt.plot(x,np.cos(x))

plt.plot(x,np.log(x))

plt.plot(x,np.exp(x))


t=np.linspace(0,2*math.pi)

x=2*np.sin(t)

y=3*np.cos(t)

plt.plot(x,y)

plt.text(0,0,r'$\frac{x^2}{2}+\frac{y^2}{3}=1$',fontsize=15)


import pandas as pd

BSdata = pd.read_csv('../data/BSdata.csv',encoding="utf-8")

plt.scatter(BSdata['身高'], BSdata['体重'], s=BSdata['支出'])


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = Axes3D(fig)

X=np.linspace(-4,4,20) #X = np.arange(-4, 4, 0.5);

Y=np.linspace(-4,4,20) #Y = np.arange(-4, 4, 0.5)

X, Y = np.meshgrid(X, Y)

Z = np.sqrt(X**2 + Y**2)

ax.plot_surface(X, Y, Z);


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(BSdata['身高'], BSdata['体重'], BSdata['支出'])


import seaborn as sns


# 绘制箱线图

sns.boxplot(x=BSdata['身高'])

# 竖着放的箱线图，也就是将x换成y

sns.boxplot(y=BSdata['身高'])

# 分组绘制箱线图，分组因子是性别，在x轴不同位置绘制

sns.boxplot(x='性别', y='身高',data=BSdata)

# 分组箱线图，分子因子是smoker，不同的因子用不同颜色区分, 相当于分组之后又分组

sns.boxplot(x='开设', y='支出',hue='性别',data=BSdata)



sns.violinplot(x='性别', y='身高',data=BSdata)

sns.violinplot(x='开设', y='支出',hue='性别',data=BSdata)


sns.stripplot(x='性别', y='身高',data=BSdata)

sns.stripplot(x='性别', y='身高',data=BSdata,jitter=True)

sns.stripplot(y='性别', x='身高',data=BSdata,jitter=True)


sns.barplot(x='性别', y='身高',data=BSdata,ci=0,palette="Blues_d")


sns.countplot(x='性别',data=BSdata)

sns.countplot(y='开设',data=BSdata)

sns.countplot(x='性别',hue="开设",data=BSdata)


sns.catplot(x='性别',col="开设", col_wrap=3,data=BSdata, kind="count", height=2.5, aspect=.8)


sns.distplot(BSdata['身高'], kde=True, bins=20, rug=True);

sns.jointplot(x='身高', y='体重', data=BSdata);

sns.pairplot(BSdata[['身高','体重','支出']]);


conda install -c conda-forge ggplot


from ggplot import *

import matplotlib.pyplot as plt              #基本绘图包


plt.rcParams['font.sans-serif']=['KaiTi'];   #SimHei黑体


plt.rcParams['axes.unicode_minus']=False;    #正常显示图中负号


qplot('身高',data=BSdata, geom='histogram')


qplot('开设',data=BSdata, geom='bar')


qplot('身高','体重',data=BSdata,color='性别')

qplot('身高','体重',data=BSdata,color='性别',size='性别')


GP=ggplot(aes(x='身高',y='体重'),data=BSdata);GP #绘制直角坐标系

GP + geom_point()                  #增加点图

GP + geom_line()                   #增加线图


ggplot(BSdata,aes(x='身高',y='体重')) + geom_point() + geom_line()

ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_bw()


ggplot(BSdata,aes(x='身高'))+ geom_histogram()


ggplot(BSdata,aes(x='身高',y='体重')) + geom_point()


ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()


ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))


ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))+geom_line(aes(y='体重'))


ggplot(BSdata,aes(x='身高',y='体重')) + geom_point() + facet_wrap('性别')


ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_bw()

