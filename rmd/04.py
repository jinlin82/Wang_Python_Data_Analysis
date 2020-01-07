import numpy as np
import matplotlib.pyplot as plt
plt.close()

import pandas as pd

BSdata = pd.read_csv('./data/BSdata.csv')

BSdata.describe()

BSdata[['性别','开设','课程','软件']].describe()


T1=BSdata.性别.value_counts();T1


T1/sum(T1)*100


BSdata.身高.mean()


BSdata.身高.median()


BSdata.身高.max()-BSdata.身高.min()


BSdata.身高.var()


BSdata.身高.std()


BSdata.身高.quantile(0.75)-BSdata.身高.quantile(0.25)


BSdata.身高.skew()


BSdata.身高.kurt()


def stats(x):
    stat=[x.count(),x.min(),x.quantile(.25),x.mean(),x.median(),
         x.quantile(.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min', 'Q1(25%)','Mean','Median',
                   'Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    return(stat)




stats(BSdata.身高)

stats(BSdata.支出)


import matplotlib.pyplot as plt              #基本绘图包

plt.rcParams['font.sans-serif']=['KaiTi'];   #SimHei黑体

plt.rcParams['axes.unicode_minus']=False;    #正常显示图中负号

# plt.figure(figsize=(6,5));                   #图形大小


X=['A','B','C','D','E','F','G']

Y=[1,4,7,3,2,5,6]

plt.bar(X,Y) # 条图





plt.pie(Y,labels=X)  # 饼图


plt.plot(X,Y)  #线图 plot


plt.hist(BSdata.身高)  # 频数直方图


plt.hist(BSdata.身高,density=True) # 频率直方图


plt.hist(BSdata.身高,density=True) # 频率直方图


plt.scatter(BSdata.身高, BSdata.体重);  # 散点图


plt.ylim(0,8);

plt.xlabel('names');plt.ylabel('values');

plt.xticks(range(len(X)), X)


plt.plot(X,Y,linestyle='--',marker='o')


plt.plot(X,Y,'o--'); plt.axvline(x=1);plt.axhline(y=4)


plt.plot(X,Y);plt.text(2,7,'peakpoint')


plt.plot(X,Y,label=u'折线')


s=[0.1,0.4,0.7,0.3,0.2,0.5,0.6]

plt.bar(X,Y,yerr=s,error_kw={'capsize':5})


'''一行绘制两个图形'''

plt.figure(figsize=(12,6));

plt.subplot(121); plt.bar(X,Y);

plt.subplot(122); plt.plot(Y)


'''一列绘制两个图形'''

plt.figure(figsize=(7,10));

plt.subplot(211); plt.bar(X,Y);

plt.subplot(212); plt.plot(Y)


'''一页绘制两个图形'''

fig,ax = plt.subplots(1,2,figsize=(14,6))

ax[0].bar(X,Y)

ax[1].plot(X,Y)


'''一页绘制四个图形'''

fig,ax=plt.subplots(2,2,figsize=(15,10))

ax[0,0].bar(X,Y); ax[0,1].pie(Y,labels=X)

ax[1,0].plot(Y); ax[1,1].plot(Y,'.-',linewidth=3)


BSdata['体重'].plot(kind='line');

BSdata['体重'].plot(kind='hist');

BSdata['体重'].plot(kind='box');


BSdata['体重'].plot(kind='density',title='Density')


BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='box')


BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='density')


BSdata[['身高','体重','支出']].plot(subplots=True,layout=(3,1),kind='density')


T1=BSdata['开设'].value_counts();T1

pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100})


T1.plot(kind='bar'); #T1.sort_values().plot(kind='bar');

T1.plot(kind='pie')


BSdata['开设'].value_counts()

#BSdata.pivot_table(values='学号',index='开设',aggfunc=len)


def tab(x,plot=False): #计数频数表

   f=x.value_counts();f

   s=sum(f);

   p=round(f/s*100,3);p

   T1=pd.concat([f,p],axis=1);

   T1.columns=['例数','构成比'];

   T2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])

   Tab=T1.append(T2)

   if plot:

     fig,ax = plt.subplots(2,1,figsize=(8,15))

     ax[0].bar(f.index,f); # 条图

     ax[1].pie(p,labels=p.index,autopct='%1.2f%%');  # 饼图

   return(round(Tab,3))


tab(BSdata.开设,True)


pd.cut(BSdata.身高,bins=10).value_counts()


pd.cut(BSdata.身高,bins=10).value_counts().plot(kind='bar')


pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()


pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar')


def freq(X,bins=10): #计量频数表与直方图

    H=plt.hist(X,bins);

    a=H[1][:-1];a

    b=H[1][1:];b

    f=H[0];f

    p=f/sum(f)*100;p

    cp=np.cumsum(p);cp

    Freq=pd.DataFrame([a,b,f,p,cp])

    Freq.index=['[下限','上限)','频数','频率(%)','累计频数(%)']

    return(round(Freq.T,2))


freq(BSdata.体重)


pd.crosstab(BSdata.开设,BSdata.课程)


pd.crosstab(BSdata.开设,BSdata.课程,margins=True)


pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index')


pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns')


pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all').round(3)


T2=pd.crosstab(BSdata.开设,BSdata.课程);T2

T2.plot(kind='bar');


T2.plot(kind='bar',stacked=True);


BSdata.groupby(['性别'])

type(BSdata.groupby(['性别']))


BSdata.groupby(['性别'])['身高'].mean()


BSdata.groupby(['性别'])['身高'].size()


BSdata.groupby(['性别','开设'])['身高'].mean()


BSdata.groupby(['性别'])['身高'].agg([np.mean, np.std])


BSdata.groupby(['性别'])['身高','体重'].apply(np.mean)


BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean)


BSdata.pivot_table(index=['性别'],values=['学号'],aggfunc=len)


BSdata.pivot_table(values=['学号'],index=['性别','开设'],aggfunc=len)


BSdata.pivot_table(values=['学号'],index=['开设'],columns=['性别'],aggfunc=len)


BSdata.pivot_table(index=['性别'],values=["身高"],aggfunc=np.mean)


BSdata.pivot_table(index=['性别'],values=["身高"],aggfunc=[np.mean,np.std])


BSdata.pivot_table(index=["性别"],values=["身高","体重"])


BSdata.pivot_table('学号', ['性别','开设'], '课程', aggfunc=len, margins=True, margins_name='合计')


BSdata.pivot_table(['身高','体重'],['性别',"开设"],aggfunc=[len,np.mean,np.std] )

