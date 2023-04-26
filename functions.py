import math
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def Gaussian_pdf(x,name,para):
    mu=para[name+'_mean']
    sigma=para[name+'_sd']
    return math.exp(((x-mu)/sigma)**2*(-1/2))/(sigma*math.sqrt(2*math.pi))

def normialise(tem_dist):
    return [tem_dist[i]/sum(tem_dist) for i in range(len(tem_dist))]
def second_moment(name,para):
    return para[name+'_mean']**2+para[name+'_sd']**2
def c_generate(n,m):
    C=np.random.random((n, m))
    for i in range(n):
        for j in range(n):
            C[i,j]=abs(i-j)
    return C

def algorithms(reg,m,n,g,f,C):
    K=np.exp(-C/C.max()/reg)
    interations=10000
    u=np.ones((n,1))
    for i in range(1,interations):
        v=g/np.dot(K.T,u)
        u=f/np.dot(K,v)
    return np.dot(np.diag(u.reshape((1,-1))[0]),np.dot(K,np.diag(v.reshape((1,-1))[0])))

def assess(m,n,g,f,C,output):
    print('sum of violation of f:',sum(abs(np.sum(output,1)-f.reshape(n))))
    print('sum of violation of g:',sum(abs(np.sum(output,0)-g.reshape(m))))
    print('total cost:',sum(sum(output*C)))
    print('entropy:',sum(sum(-output*np.log(output+0.1**3))))
    print('============================================')
    
def plots(x_range,g,f,output):
    fig = plt.figure(figsize=(4, 3))
    #gs = fig.add_gridspec(2, 2, width_ratios=(bin, 1), height_ratios=(1, bin),left=0.1, right=0.9, bottom=0.1, top=0.9,wspace=0.05, hspace=0.05)
    # Create the Axes.
    #ax = fig.add_subplot(gs[1, 0])
    ax.pcolormesh(x_range, x_range, output, cmap='Blues')
    # ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    # ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # ax_histx.tick_params(axis="x", labelbottom=False)
    # ax_histy.tick_params(axis="y", labelleft=False)
    # ax_histx.plot(x_range,g)
    # ax_histy.plot(f,x_range) 

def OT(reg,x_range,g,f):
    m=len(g)
    n=len(f)
    C=c_generate(n,m)
    output=algorithms(reg,m,n,g.reshape(m,-1),f.reshape(n,-1),C)
    assess(m,n,g,f,C,output)
    plots(x_range,g,f,output)
    return output

def data_generation(num,para):
    name_list=['af','bf','am','bm']
    df=pd.DataFrame(columns=['X', 'U', 'S'])
    for name in name_list:
        size=int(para[name]*num)
        X=np.floor((np.random.normal(para['X_'+name+'_mean'],para['X_'+name+'_sd'],size=[size])-horizen[0])/width)
        #np.random.normal(para['X_'+name+'_mean'],para['X_'+name+'_sd'],size=[size])
        U=[name[0]]*size
        S=[name[1]]*size
        df=pd.concat([df, pd.DataFrame([X,U,S], index=['X','U','S']).T], ignore_index=True)
    df['W']=1
    df['X']=df['X'].astype('int64')  
    return df
def empirical_distribution(sub,dist):
    bin=dist['bin']
    dist=np.zeros(bin)
    for i in range(bin):
        subset=sub[sub['X']==i] #bin_value=x_range[i] #sub[(sub['X']>=bin_value)&(sub['X']<bin_value+width)]
        if subset.shape[0]>0:
            dist[i]=sum(subset['W'])
    if sum(dist)>0:
        return dist/sum(dist)
    else:
        return dist

def samples_groupby(data):
    # for better complexity
    df=data.groupby(by=['X','U','S'],as_index=False).sum()
    return df[df['W']!=0]
def projection(df,coupling):
    df_t=pd.DataFrame(columns=['X', 'U', 'S','W'])
    for i in range(df.shape[0]):
        orig=df.iloc[i]
        sub=pd.DataFrame(columns=['X','W'],index=[*range(bin)])
        sub['X']=[*range(bin)]
        sub['W']=coupling[:,orig[0]]/(sum(coupling[:,orig[0]])+0.0001)*orig[3]
        sub['U']=orig[1]
        sub['S']=orig[2]
        df_t=pd.concat([df_t, samples_groupby(sub)], ignore_index=True)
    return df_t