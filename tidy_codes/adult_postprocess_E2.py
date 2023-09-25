import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from functions import*

K=200
e=0.01

var_list=['hoursperweek','age','capitalgain','capitalloss' ,'education-num'] #
var_dim=len(var_list)
pa='sex'
pa_dict={'Male':1,'Female':0,'White':1,'Black':0}

messydata=pd.read_csv('/home/zhouqua1/OT_Debiasing/data/adult_csv.csv',usecols=var_list+[pa,'class'])
messydata=messydata.rename(columns={pa:'S','class':'Y'})
messydata['S']=messydata['S'].replace(pa_dict)
messydata['Y']=messydata['Y'].replace({'>50K':1,'<=50K':0})
messydata=messydata[(messydata['S']==1)|(messydata['S']==0)]
for col in var_list+['S','Y']:
    messydata[col]=messydata[col].astype('category')
messydata['W']=1

X=messydata[var_list+['S','W']].to_numpy() # [X,S,W]
y=messydata['Y'].to_numpy() #[Y]

tv_dist=dict()
for x_name in var_list:
    x_range_single=list(pd.pivot_table(messydata,index=x_name,values=['W'])[('W')].index) 
    dist=rdata_analysis(messydata,x_range_single,x_name)
    tv_dist[x_name]=sum(abs(dist['x_0']-dist['x_1']))/2
x_list=[]
for key,val in tv_dist.items():
    if val>0.08:
        x_list+=[key]

report=pd.DataFrame(columns=['DI','f1 macro','f1 micro','f1 weighted','TV distance','method'])
for ignore in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf=RandomForestClassifier(max_depth=5, random_state=0).fit(X_train[:,0:var_dim],y_train)

    df_test=pd.DataFrame(np.concatenate((X_test,y_test.reshape(-1,1)), axis=1),columns=var_list+['S','W','Y'])
    df_test=df_test.groupby(by=var_list+['S','Y'],as_index=False).sum()

    if len(x_list)>1:
        df_test['X']=[tuple(df_test[x_list].values[r]) for r in range(df_test.shape[0])]
        x_range=list(set(df_test['X']))
        weight=list(1/(df_test[x_list].max()-df_test[x_list].min())) # because 'education-num' range from 1 to 16 while others 1 to 4
        C=c_generate_higher(x_range,weight)
    else:
        df_test['X']=df_test[x_list]
        x_range=list(set(df_test['X']))
        C=c_generate(x_range)

    bin=len(x_range)
    var_range=list(pd.pivot_table(df_test,index=var_list,values=['S','W','Y']).index)
    dist=rdata_analysis(df_test,x_range,'X')
    dist['t_x']=dist['x'] # #dist['x'] #dist['x_0']*0.5+dist['x_1']*0.5 
    dist['v']=[(dist['x_0'][i]-dist['x_1'][i])/dist['x'][i] for i in range(bin)]
    px=np.matrix(dist['x']).T
    ptx=np.matrix(dist['t_x']).T
    if np.any(dist['x_0']==0): 
        p0=np.matrix((dist['x_0']+1.0e-9)/sum(dist['x_0']+1.0e-9)).T
    else:
        p0=np.matrix(dist['x_0']).T 
    if np.any(dist['x_1']==0):
        p1=np.matrix((dist['x_1']+1.0e-9)/sum(dist['x_1']+1.0e-9)).T
    else:
        p1=np.matrix(dist['x_1']).T 
    V=np.matrix(dist['v']).T

    coupling_base=baseline(C,e,px,ptx,V,K)
    coupling_bary=baseline(C,e,p0,p1,V,K)
    coupling_part2=partial_repair(C,e,px,ptx,V,1.0e-2,K)
    coupling_part3=partial_repair(C,e,px,ptx,V,1.0e-3,K)
    #coupling_part4=partial_repair(C,e,px,ptx,V,5.0e-4,K)[K*3]

    tv_base=assess_tv(df_test,coupling_base,x_range,x_list,var_list)
    tv_part2=assess_tv(df_test,coupling_part2,x_range,x_list,var_list)
    tv_part3=assess_tv(df_test,coupling_part3,x_range,x_list,var_list)
    #tv_part4=assess_tv(df_test,coupling_part4,x_range,x_list,var_list)

    y_pred=clf.predict(np.array(df_test[var_list]))
    y_pred_base=postprocess(df_test,coupling_base,x_list,x_range,var_list,var_range,clf)
    y_pred_bary,tv_bary=postprocess_bary(df_test,coupling_bary,x_list,x_range,var_list,var_range,clf)
    y_pred_part2=postprocess(df_test,coupling_part2,x_list,x_range,var_list,var_range,clf)
    y_pred_part3=postprocess(df_test,coupling_part3,x_list,x_range,var_list,var_range,clf)
    #y_pred_part4=postprocess(df_test,coupling_part4,x_list,x_range,var_list,var_range,clf)

    new_row=pd.Series({'DI':DisparateImpact_postprocess(df_test,y_pred),
                        'f1 macro':f1_score(df_test['Y'], y_pred, average='macro',sample_weight=df_test['W']),
                        'f1 micro':f1_score(df_test['Y'], y_pred, average='micro',sample_weight=df_test['W']),
                        'f1 weighted':f1_score(df_test['Y'], y_pred, average='weighted',sample_weight=df_test['W']),
                        'TV distance':sum(abs(dist['x_0']-dist['x_1']))/2,'method':'origin'})
    new_row_base=pd.Series({'DI':DisparateImpact_postprocess(df_test,y_pred_base),
                        'f1 macro':f1_score(df_test['Y'], y_pred_base, average='macro',sample_weight=df_test['W']),
                        'f1 micro':f1_score(df_test['Y'], y_pred_base, average='micro',sample_weight=df_test['W']),
                        'f1 weighted':f1_score(df_test['Y'], y_pred_base, average='weighted',sample_weight=df_test['W']),
                        'TV distance':tv_base,'method':'baseline'})
    new_row_bary=pd.Series({'DI':DisparateImpact_postprocess(df_test,y_pred_bary),
                        'f1 macro':f1_score(df_test['Y'], y_pred_bary, average='macro',sample_weight=df_test['W']),
                        'f1 micro':f1_score(df_test['Y'], y_pred_bary, average='micro',sample_weight=df_test['W']),
                        'f1 weighted':f1_score(df_test['Y'], y_pred_bary, average='weighted',sample_weight=df_test['W']),
                        'TV distance':tv_bary,'method':'barycentre'})
    new_row_part2=pd.Series({'DI':DisparateImpact_postprocess(df_test,y_pred_part2),
                        'f1 macro':f1_score(df_test['Y'], y_pred_part2, average='macro',sample_weight=df_test['W']),
                        'f1 micro':f1_score(df_test['Y'], y_pred_part2, average='micro',sample_weight=df_test['W']),
                        'f1 weighted':f1_score(df_test['Y'], y_pred_part2, average='weighted',sample_weight=df_test['W']),
                        'TV distance':tv_part2,'method':'partial repair2'})
    new_row_part3=pd.Series({'DI':DisparateImpact_postprocess(df_test,y_pred_part3),
                        'f1 macro':f1_score(df_test['Y'], y_pred_part3, average='macro',sample_weight=df_test['W']),
                        'f1 micro':f1_score(df_test['Y'], y_pred_part3, average='micro',sample_weight=df_test['W']),
                        'f1 weighted':f1_score(df_test['Y'], y_pred_part3, average='weighted',sample_weight=df_test['W']),
                        'TV distance':tv_part3,'method':'partial repair3'})
    # new_row_part4=pd.Series({'DI':DisparateImpact_postprocess(df_test,y_pred_part4),
    #                     'f1 macro':f1_score(df_test['Y'], y_pred_part4, average='macro',sample_weight=df_test['W']),
    #                     'f1 micro':f1_score(df_test['Y'], y_pred_part4, average='micro',sample_weight=df_test['W']),
    #                     'f1 weighted':f1_score(df_test['Y'], y_pred_part4, average='weighted',sample_weight=df_test['W']),
    #                     'TV distance':tv_part4,'method':'partial repair4'})

    #report = pd.concat([report,new_row.to_frame().T,new_row_base.to_frame().T,new_row_part2.to_frame().T,new_row_part3.to_frame().T,new_row_part4.to_frame().T], ignore_index=True) #,new_row_part4.to_frame().T
    report = pd.concat([report,new_row.to_frame().T,new_row_base.to_frame().T,new_row_bary.to_frame().T,new_row_part2.to_frame().T,new_row_part3.to_frame().T], ignore_index=True) #,new_row_part4.to_frame().T
    #report = pd.concat([report,new_row.to_frame().T,new_row_base.to_frame().T,new_row_bary.to_frame().T,new_row_part2.to_frame().T,new_row_part3.to_frame().T,new_row_part4.to_frame().T], ignore_index=True) #,new_row_part4.to_frame().T
    
report.to_csv('/home/zhouqua1/OT_Debiasing/data/report_postprocess_bary'+str(pa)+'.csv',index=None)