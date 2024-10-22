#import math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#from itertools import chain

def normialise(tem_dist):
    return [tem_dist[i]/sum(tem_dist) for i in range(len(tem_dist))]

def tmp_generator(gamma_dict,num,q_dict,q_num,L):
    bin=gamma_dict[0].shape[0]
    if q_num<=0:
        q=np.matrix(np.ones((bin,bin)))
    else:
        q=q_dict[q_num]
    tmp_gamma=np.zeros((bin,bin))
    tmp_q=np.zeros((bin,bin))
    for i in range(bin):
        for j in range(bin):
            if gamma_dict[num-L].item(i,j) != 0:
                tmp_gamma[i,j]=q.item(i,j)*gamma_dict[num-1].item(i,j)*gamma_dict[num-L-1].item(i,j)/gamma_dict[num-L].item(i,j)
                tmp_q[i,j]=q.item(i,j)*gamma_dict[num-L-1].item(i,j)/gamma_dict[num-L].item(i,j)
            else:
                # to avoid zero division error
                tmp_gamma[i,j]=q.item(i,j)*gamma_dict[num-1].item(i,j)*gamma_dict[num-L-1].item(i,j)/(1.0e-9) 
                tmp_q[i,j]=q.item(i,j)*gamma_dict[num-L-1].item(i,j)/(1.0e-9)
    return np.matrix(tmp_gamma),np.matrix(tmp_q)     

def newton(fun,dfun,a, stepmax, tol):
    if abs(fun(a))<=tol: return a
    for step in range(1, stepmax+1):
        b=a-fun(a)/dfun(a)
        if abs(fun(b))<=tol:
            return b
        else:
            a = b
    return b 

# simplist
def baseline(C,e,px,ptx,V,K):
    # V is not used
    bin=len(px)
    bbm1=np.matrix(np.ones(bin)).T
    #I=np.where(~(V==0))[0].tolist()
    xi=np.exp(-C/e)
    gamma_classic=dict()
    gamma_classic[0]=np.matrix(xi+1.0e-9)
    for repeat in range(K):
        gamma_classic[1+2*repeat]=np.matrix(np.diag((px/(gamma_classic[2*repeat] @ bbm1)).A1))@gamma_classic[2*repeat] #np.diag(dist['x']/sum(gamma_classic.T))@gamma_classic
        gamma_classic[2+2*repeat]=gamma_classic[1+2*repeat]@np.matrix(np.diag((ptx/(gamma_classic[1+2*repeat].T @ bbm1)).A1))

    #assess(bin,dist['x'],dist['t_x'],C,V,gamma_classic[2*K])
    return gamma_classic[2*K]

# our method | total repair
def total_repair(C,e,px,ptx,V,K):
    bin=len(px)
    bbm1=np.matrix(np.ones(bin)).T
    I=np.where(~(V==0))[0].tolist()
    xi=np.exp(-C/e)
    gamma_dict=dict()
    gamma_dict[0]=np.matrix(xi+1.0e-9)
    gamma_dict[1]=np.matrix(np.diag((px/(gamma_dict[0] @ bbm1)).A1))@gamma_dict[0]
    gamma_dict[2]=gamma_dict[1]@np.matrix(np.diag((ptx/(gamma_dict[1].T @ bbm1)).A1))
    # step 3
    J=np.where(~((gamma_dict[2].T @ V).A1 ==0))[0].tolist()
    nu=np.zeros(bin)
    gamma_dict[3]=np.copy(gamma_dict[2])
    for j in J:
        fun = lambda z: sum(gamma_dict[2].item(i,j)*V.item(i)*np.exp(z*V.item(i)) for i in I)
        dfun = lambda z: sum(gamma_dict[2].item(i,j)*(V.item(i))**2*np.exp(z*V.item(i)) for i in I)
        nu = newton(fun,dfun,0,stepmax = 50,tol = 1.0e-5) 
        for i in I:
            gamma_dict[3][i,j]=np.exp(nu*V.item(i))*gamma_dict[2].item(i,j)
    gamma_dict[3]=np.matrix(gamma_dict[3])

    #=========================
    L=3
    q_dict=dict()
    for loop in range(1,K):
        tmp,q_dict[(loop-1)*L+1]=tmp_generator(gamma_dict,loop*L+1,q_dict,(loop-2)*L+1,L) #np.matrix(gamma_dict[3].A1*gamma_dict[0].A1/gamma_dict[1].A1)
        gamma_dict[loop*L+1]=np.matrix(np.diag((px/(tmp @ bbm1)).A1))@tmp

        tmp,q_dict[(loop-1)*L+2]=tmp_generator(gamma_dict,loop*L+2,q_dict,(loop-2)*L+2,L)  #np.matrix(gamma_dict[4].A1*gamma_dict[1].A1/gamma_dict[2].A1)
        gamma_dict[loop*L+2]=tmp@np.matrix(np.diag((ptx/(tmp.T @ bbm1)).A1))

        # step 3
        tmp,q_dict[(loop-1)*L+3]=tmp_generator(gamma_dict,loop*L+3,q_dict,(loop-2)*L+3,L)  #np.matrix(gamma_dict[5].A1*gamma_dict[2].A1/gamma_dict[3].A1)
        J=np.where(~((abs(np.matrix(tmp).T @ V).A1)<=1.0e-4))[0].tolist()
        gamma_dict[loop*L+3]=np.copy(tmp)
        for j in J:
            fun = lambda z: sum(tmp.item(i,j)*V.item(i)*np.exp(z*V.item(i)) for i in I)
            dfun = lambda z: sum(tmp.item(i,j)*(V.item(i))**2*np.exp(z*V.item(i)) for i in I)
            nu = newton(fun,dfun,0,stepmax = 50,tol = 1.0e-5) 
            for i in I:
                gamma_dict[loop*L+3][i,j]=np.exp(nu*V.item(i))*tmp.item(i,j)
        gamma_dict[loop*L+3]=np.matrix(gamma_dict[loop*L+3])

        if sum(abs(gamma_dict[loop*L+3].T@V))<=1.0e-5:  #'tr violation:'
                break;
    #assess(bin,dist['x'],dist['t_x'],C,V,gamma_dict[K*L])
    return gamma_dict[loop*L+3]

# our method | partial repair
def partial_repair(C,e,px,ptx,V,theta_scale,K):
    bin=len(px)
    bbm1=np.matrix(np.ones(bin)).T
    I=np.where(~(V==0))[0].tolist()
    xi=np.exp(-C/e)
    theta=bbm1*theta_scale
    gamma_dict=dict()
    gamma_dict[0]=np.matrix(xi+1.0e-9)
    gamma_dict[1]=np.matrix(np.diag((px/(gamma_dict[0] @ bbm1)).A1))@gamma_dict[0]
    gamma_dict[2]=gamma_dict[1]@np.matrix(np.diag((ptx/(gamma_dict[1].T @ bbm1)).A1))
    # step 3
    Jplus=np.where(~((gamma_dict[2].T @ V).A1 <=theta.A1))[0].tolist()
    Jminus=np.where(~((gamma_dict[2].T @ V).A1>=-theta.A1))[0].tolist()
    gamma_dict[3]=np.copy(gamma_dict[2])
    for j in Jplus:
        fun = lambda z: sum(gamma_dict[2].item(i,j)*V.item(i)*np.exp(-z*V.item(i)) for i in I)-theta.item(j)
        dfun = lambda z: -sum(gamma_dict[2].item(i,j)*(V.item(i))**2*np.exp(-z*V.item(i)) for i in I)
        nu = newton(fun,dfun,0.5,stepmax = 50,tol = 1.0e-5)
        for i in I:
            gamma_dict[3][i,j]=np.exp(-nu*V.item(i))*gamma_dict[2].item(i,j)
    for j in Jminus:
        fun = lambda z: sum(gamma_dict[2].item(i,j)*V.item(i)*np.exp(-z*V.item(i)) for i in I)+theta.item(j)
        dfun = lambda z: -sum(gamma_dict[2].item(i,j)*(V.item(i))**2*np.exp(-z*V.item(i)) for i in I)
        nu = newton(fun,dfun,0,stepmax = 50,tol = 1.0e-5)
        for i in I:
            gamma_dict[3][i,j]=np.exp(-nu*V.item(i))*gamma_dict[2].item(i,j)
    gamma_dict[3]=np.matrix(gamma_dict[3])

    #=========================
    L=3
    q_dict=dict()
    for loop in range(1,K):
        tmp,q_dict[(loop-1)*L+1]=tmp_generator(gamma_dict,loop*L+1,q_dict,(loop-2)*L+1,L) #np.matrix(gamma_dict[3].A1*gamma_dict[0].A1/gamma_dict[1].A1)
        gamma_dict[loop*L+1]=np.matrix(np.diag((px/(tmp @ bbm1)).A1))@tmp

        tmp,q_dict[(loop-1)*L+2]=tmp_generator(gamma_dict,loop*L+2,q_dict,(loop-2)*L+2,L)  #np.matrix(gamma_dict[4].A1*gamma_dict[1].A1/gamma_dict[2].A1)
        gamma_dict[loop*L+2]=tmp@np.matrix(np.diag((ptx/(tmp.T @ bbm1)).A1))

        # step 3
        tmp,q_dict[(loop-1)*L+3]=tmp_generator(gamma_dict,loop*L+3,q_dict,(loop-2)*L+3,L)  #np.matrix(gamma_dict[5].A1*gamma_dict[2].A1/gamma_dict[3].A1)
        Jplus=np.where(~((np.matrix(tmp).T @ V).A1 <=theta.A1))[0].tolist()
        Jminus=np.where(~((np.matrix(tmp).T @ V).A1>=-theta.A1))[0].tolist()
        gamma_dict[loop*L+3]=np.copy(tmp)
        for j in Jplus:
            fun = lambda z: sum(tmp.item(i,j)*V.item(i)*np.exp(-z*V.item(i)) for i in I)-theta.item(j)
            dfun = lambda z: -sum(tmp.item(i,j)*(V.item(i))**2*np.exp(-z*V.item(i)) for i in I)
            nu = newton(fun,dfun,0,stepmax = 50,tol = 1.0e-5) 
            for i in I:
                gamma_dict[loop*L+3][i,j]=np.exp(-nu*V.item(i))*tmp.item(i,j)
        for j in Jminus:
            fun = lambda z: sum(tmp.item(i,j)*V.item(i)*np.exp(-z*V.item(i)) for i in I)+theta.item(j)
            dfun = lambda z: -sum(tmp.item(i,j)*(V.item(i))**2*np.exp(-z*V.item(i)) for i in I)
            nu = newton(fun,dfun,0,stepmax = 50,tol = 1.0e-5) 
            for i in I:
                gamma_dict[loop*L+3][i,j]=np.exp(-nu*V.item(i))*tmp.item(i,j)
        gamma_dict[loop*L+3]=np.matrix(gamma_dict[loop*L+3])
        if sum(abs(gamma_dict[loop*L+3].T@V))<=1.0e-5:  #'tr violation:'
            break;
    return gamma_dict[loop*L+3]

def empirical_distribution(sub,x_range):
    bin=len(x_range)
    distrition=np.zeros(bin)
    for i in range(bin):
        subset=sub[sub['X']==x_range[i]] #bin_value=x_range[i] #sub[(sub['X']>=bin_value)&(sub['X']<bin_value+width)]
        if subset.shape[0]>0:
            distrition[i]=sum(subset['W'])
    if sum(distrition)>0:
        return distrition/sum(distrition)
    else:
        return distrition

def DisparateImpact(X_test,y_pred):
    dim=X_test.shape[1]-2
    df_test=pd.DataFrame(np.concatenate((X_test,y_pred.reshape(-1,1)), axis=1),columns=[*range(dim)]+['S','W','f'])
    numerator=sum(df_test[(df_test['S']==0)&(df_test['f']==1)]['W'])/sum(df_test[df_test['S']==0]['W'])
    denominator=sum(df_test[(df_test['S']==1)&(df_test['f']==1)]['W'])/sum(df_test[df_test['S']==1]['W'])
    return numerator/denominator
    
def rdata_analysis(rdata,x_range,x_name):
    rdist=dict()
    pivot=pd.pivot_table(rdata,index=x_name,values=['W'],aggfunc=[np.sum])[('sum','W')]
    rdist['x']= np.array([pivot[i] for i in x_range])/sum([pivot[i] for i in x_range]) #empirical_distribution(rdata,x_range)
    if rdata[rdata['S']==0].shape[0]>0:
        pivot0=pd.pivot_table(rdata[rdata['S']==0],index=x_name,values=['W'],aggfunc=[np.sum])[('sum','W')]
        rdist['x_0']=np.array([pivot0[i] if i in list(pivot0.index) else 0 for i in x_range])/sum([pivot0[i] if i in list(pivot0.index) else 0 for i in x_range]) #empirical_distribution(rdata[rdata['S']==0],x_range)
    if rdata[rdata['S']==1].shape[0]>0:
        pivot1=pd.pivot_table(rdata[rdata['S']==1],index=x_name,values=['W'],aggfunc=[np.sum])[('sum','W')]
        rdist['x_1']=np.array([pivot1[i] if i in list(pivot1.index) else 0 for i in x_range])/sum([pivot1[i] if i in list(pivot1.index) else 0 for i in x_range]) #empirical_distribution(rdata[rdata['S']==1],x_range)
    return rdist

def c_generate(x_range):
    bin=len(x_range)
    C=np.random.random((bin,bin))
    for i in range(bin):
        for j in range(bin):
            C[i,j]=abs(x_range[i]-x_range[j]) 
    return C

def projection(df,coupling_matrix,x_range,x_name,var_list):
    bin=len(x_range)
    var_list_tmp=var_list[:]
    var_list_tmp.remove(x_name)
    var_list_tmp=[x_name]+var_list_tmp # place the var that needs to be repaired the first
    df=df[var_list_tmp+['S','W','Y']]
    coupling=coupling_matrix.A1.reshape((bin,bin))
    df_t=pd.DataFrame(columns=var_list_tmp+['S','W','Y'])
    for i in range(df.shape[0]):
        orig=df.iloc[i]
        loc=np.where([x_range[i]==orig[x_name] for i in range(bin)])[0][0]
        rows=np.nonzero(coupling[loc,:])[0]
        sub_dict={x_name:[x_range[r] for r in rows],'W':list(coupling[loc,rows]/(sum(coupling[loc,rows]))*orig['W'])}
        sub_dict.update({var:[orig[var]]*len(rows) for var in var_list_tmp[1:]+['S','Y']})
        sub=pd.DataFrame(data=sub_dict, index=rows)
        df_t=pd.concat([df_t,sub],ignore_index=True)#pd.concat([df_t,samples_groupby(sub,x_list)], ignore_index=True)
    df_t=df_t.groupby(by=list(chain(*[var_list,'S','Y'])),as_index=False).sum()
    df_t=df_t[var_list+['S','W','Y']]
    return df_t

def c_generate_higher(x_range,weight):
    bin=len(x_range)
    dim=len(x_range[0])
    C=np.random.random((bin,bin))
    for i in range(bin):
        for j in range(bin):
            C[i,j]=sum(weight[d]*abs(x_range[i][d]-x_range[j][d]) for d in range(dim))
    return C

def projection_higher(df,coupling_matrix,x_range,x_list,var_list):
    df=df.drop(columns=x_list)
    bin=len(x_range)
    arg_list=[elem for elem in var_list if elem not in x_list]
    df=df[arg_list+['X','S','W','Y']]
    coupling=coupling_matrix.A1.reshape((bin,bin))
    df_t=pd.DataFrame(columns=arg_list+['X','S','W','Y'])
    for i in range(df.shape[0]):
        orig=df.iloc[i]
        loc=np.where([x_range[b]==orig['X'] for b in range(bin)])[0][0]
        #rows=np.nonzero(coupling[loc,:])[0]
        sub_dict={'X':x_range,'W':list(coupling[loc,:]/(sum(coupling[loc,:]))*orig['W'])}
        sub_dict.update({var:[orig[var]]*bin for var in arg_list+['S','Y']})
        sub=pd.DataFrame(data=sub_dict, index=[*range(bin)])
        df_t=pd.concat([df_t,sub],ignore_index=True) #pd.concat([df_t,samples_groupby(sub,x_list)], ignore_index=True)
    return df_t

# def projection_higher(df,coupling_matrix,x_range,x_list,var_list):
#     df=df.drop(columns=x_list)
#     bin=len(x_range)
#     dim=len(x_list)
#     arg_list=[elem for elem in var_list if elem not in x_list]
#     df=df[arg_list+['X','S','W','Y']]
#     coupling=coupling_matrix.A1.reshape((bin,bin))
#     df_t=pd.DataFrame(columns=arg_list+['X','S','W','Y'])
#     for i in range(df.shape[0]):
#         orig=df.iloc[i]
#         loc=np.where([x_range[i]==orig['X'] for i in range(bin)])[0][0]
#         rows=np.nonzero(coupling[loc,:])[0]
#         sub_dict={'X':[x_range[r] for r in rows],'W':list(coupling[loc,rows]/(sum(coupling[loc,rows]))*orig['W'])}
#         sub_dict.update({var:[orig[var]]*len(rows) for var in arg_list+['S','Y']})
#         sub=pd.DataFrame(data=sub_dict, index=rows)
#         df_t=pd.concat([df_t,sub],ignore_index=True)#pd.concat([df_t,samples_groupby(sub,x_list)], ignore_index=True)
#     df_t=df_t.groupby(by=list(chain(*[arg_list,'X','S','Y'])),as_index=False).sum()
#     for d in range(dim):
#         df_t[x_list[d]]=[df_t['X'][r][d] for r in range(df_t.shape[0])]
#     return df_t[var_list+['S','W','Y']]

# def projection_higher_wlabel(df,coupling_matrix,x_range,x_list,var_list):
#     dim=len(x_list)
#     #df=df.drop(columns=x_list)
#     bin=len(x_range)
#     arg_list=[elem for elem in var_list if elem not in x_list]
#     df=df[arg_list+['X','S','W']]
#     coupling=coupling_matrix.A1.reshape((bin,bin))
#     df_t=pd.DataFrame(columns=arg_list+['X','S','W'])
#     for i in range(df.shape[0]):
#         orig=df.iloc[i]
#         loc=np.where([x_range[b]==orig['X'] for b in range(bin)])[0][0]
#         rows=np.nonzero(coupling[loc,:])[0]
#         sub_dict={'X':[x_range[r] for r in rows],'W':list(coupling[loc,rows]/(sum(coupling[loc,rows]))*orig['W'])}
#         sub_dict.update({var:[orig[var]]*len(rows) for var in arg_list+['S']})
#         sub=pd.DataFrame(data=sub_dict, index=rows)
#         df_t=pd.concat([df_t,sub],ignore_index=True)#pd.concat([df_t,samples_groupby(sub,x_list)], ignore_index=True)
#     df_t=df_t.groupby(by=list(chain(*[arg_list,'X','S'])),as_index=False).sum()
#     for d in range(dim):
#         df_t[x_list[d]]=[df_t['X'][r][d] for r in range(df_t.shape[0])]
#     return df_t[var_list+['S','W']]

def postprocess(df,coupling_matrix,x_list,x_range,var_list,var_range,clf):
    dim=len(x_list)
    var_dim=len(var_list)
    bin=len(x_range)
    x_loc_dict=dict(zip(x_range,[*range(bin)]))
    arg_list=[elem for elem in var_list if elem not in x_list]
    coupling=coupling_matrix.A1.reshape((bin,bin))
    pred_repaired=dict()
    for i in range(len(var_range)):
        if var_dim>1:
            var_tmp=pd.Series({var_list[d]:var_range[i][d] for d in range(var_dim)})
            if dim>1:
                loc=x_loc_dict[tuple(var_tmp[x_list])]
            else:
                loc=x_loc_dict[var_tmp[x_list[0]]]
        else:
            var_tmp=pd.Series({var_list[0]:var_range[i]})
            loc=x_loc_dict[var_tmp[x_list[0]]]
        sub=pd.DataFrame(x_range,columns=x_list)
        for arg in arg_list:
            sub[arg]=var_tmp[arg] 
        sub=sub[var_list]
        totalweight=sum(coupling[loc,:])
        pred=int(sum(coupling[loc,:]/totalweight*clf.predict(np.array(sub).reshape(-1,var_dim)))>0.1)
        pred_repaired.update({var_range[i]:pred})
        # prob=clf.predict_log_proba(np.array(sub).reshape(-1,var_dim)) #log is better
        # prob0=sum(prob[:,0]*coupling[loc,:]/totalweight)
        # prob1=sum(prob[:,1]*coupling[loc,:]/totalweight)
        # pred_repaired.update({var_range[i]:int(prob0<prob1)})
    if var_dim>1:
        return np.array([pred_repaired[tuple(df[var_list].iloc[i])] for i in range(df.shape[0])])
    else:
        return np.array([pred_repaired[df[var_list[0]].iloc[i]] for i in range(df.shape[0])])

def DisparateImpact_postprocess(df_test,y_pred_tmp):
    df_test_tmp=df_test[:]
    df_test_tmp.insert(loc=0, column='f', value=y_pred_tmp)
    numerator=sum(df_test_tmp[(df_test_tmp['S']==0)&(df_test_tmp['f']==1)]['W'])/sum(df_test_tmp[df_test_tmp['S']==0]['W'])
    denominator=sum(df_test_tmp[(df_test_tmp['S']==1)&(df_test_tmp['f']==1)]['W'])/sum(df_test_tmp[df_test_tmp['S']==1]['W'])
    # if numerator==denominator: # to avoid zero division error
    #     return 1
    return numerator/denominator

def postprocess_bary(df,coupling_bary_matrix,x_list,x_range,var_list,var_range,clf):
    bin=len(x_range)
    coupling_bary=coupling_bary_matrix.A1.reshape((bin,bin))
    s0=df[df['S']==0]
    s1=df[df['S']==1]
    pi0=s0.shape[0]/df.shape[0]
    pi1=s1.shape[0]/df.shape[0]
    coupling0=np.zeros((bin,bin))
    coupling1=np.zeros((bin,bin))
    for i in range(bin):
        for j in range(bin):
            # assume the distance between every two adjacent x indices is the same
            ind0=int(pi0*i+pi1*j)
            ind1=int(pi0*j+pi1*i)
            coupling0[i,ind0]+=coupling_bary[i,j]
            coupling1[i,ind1]+=coupling_bary[j,i]

    # assess if dist['td{x}_0']==dist['td{x}_1']
    projectedDist_s0=rdata_analysis(projection_higher(s0,np.matrix(coupling0),x_range,x_list,var_list),x_range,'X')['x_0']
    projectedDist_s1=rdata_analysis(projection_higher(s1,np.matrix(coupling1),x_range,x_list,var_list),x_range,'X')['x_1']
    # print('tv distance between projected S-wise distributions',sum(abs(projectedDist_s0-projectedDist_s1))/2)

    s0.insert(loc=0, column='f', value=postprocess(s0,np.matrix(coupling0),x_list,x_range,var_list,var_range,clf))
    s1.insert(loc=0, column='f', value=postprocess(s1,np.matrix(coupling1),x_list,x_range,var_list,var_range,clf))
    s_concate=pd.concat([s0,s1], ignore_index=False)
    s_concate.sort_index()
    return np.array(s_concate['f']),sum(abs(projectedDist_s0-projectedDist_s1))/2

def assess_tv(df,coupling_matrix,x_range,x_list,var_list):
    df_project=projection_higher(df,coupling_matrix,x_range,x_list,var_list)
    rdist=rdata_analysis(df_project,x_range,'X')
    return sum(abs(rdist['x_0']-rdist['x_1']))/2