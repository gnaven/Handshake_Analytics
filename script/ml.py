import numpy as np
import pandas as pd
#from IPython.display import display
import matplotlib.pyplot as plt

pd.options.display.max_rows = None
pd.options.display.max_columns = None
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import metrics

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#------------------------------------------------------------

def ml(dfin, models = None, y_col = None, weight_plot=0):
    """ Runs either logistic regression or KNN through sklearn using both 
        dev and test set. The hyperparameter is set using a dev set. Post
        hyperparameter training uses train and dev set for training.
        
        :df     -pandas df - must have a 'y' column that is bool, all other
                 columns will be cast to float and used as features.        
        :model  -'LOGISTIC' or 'KNN'
        """

    df = pd.DataFrame(dfin, dtype=float)
    assert('y' in df.columns)
    df = df.dropna()

    #making the dataset balanced
    # df_1 = df[df['y']==1]
    # size_1 = df_1.shape[0]

    # df_0 = df[df['y']==0].sample(size_1)

    # df = pd.concat([df_0,df_1])

    #size of final dataset
    print('dataset shape ', df.shape)
    print('class proportion ', df['y'].mean())
    
    if df.shape[0] < 20:
        N_FOLDS = 2
    else:
        N_FOLDS = 10
    ACC_THRESH = 0.01 # dev set accuracy must be x% better to use new param   

    if models is None:
        models = ['KNN','LOGISTIC','TREE']
    
    for model in models:
        print('\nMODEL: ', model)

        if model == 'LOGISTIC':
            c_l = [0.01,0.03, 0.1, 0.3, 1,3,10,30,100,300,1000,3000,10000]
        elif model == 'KNN':       
            c_l = [50,40,35,30,25,20,18,15]
        else:
            c_l = [3,4,5,6,7,8]
        regularizer = 'l1'
            
        X_nd = df.drop('y',axis=1).values
        X_nd = normalize(X_nd) # magnitude has useful info?
        y_n = df['y'].values.astype(bool)
        skf = StratifiedKFold(shuffle = True, n_splits=N_FOLDS)
    
        acc_test_a = np.zeros(N_FOLDS)
        acc_train_a = np.zeros(N_FOLDS)

        f1_test_a = np.zeros(N_FOLDS)
        f1_train_a = np.zeros(N_FOLDS)
        
        ncol = len(X_nd[0])
        weight_list = [0]* ncol          
        
        for i, (train, test) in enumerate(skf.split(X_nd,y_n)):

            
            train_n = len(train)
            dev = train[:int(train_n/4)]  # empirically found that dev 1/4 is good
            sub_train = train[int(train_n/4):] # this is temporary train set
            best_acc = 0
            best_c = None
            # in this loop we find best hyper parameter for this split
            for c in c_l:
                if model == 'LOGISTIC': 
                    clf = linear_model.LogisticRegression(penalty=regularizer,C=c)
                elif model == 'KNN':           
                    clf = KNeighborsClassifier(n_neighbors=c, metric='euclidean',weights='uniform')
                else:
                    clf = tree.DecisionTreeClassifier(max_leaf_nodes=c)
                clf.fit(X_nd[sub_train], y_n[sub_train])
                y_pred = clf.predict(X_nd[dev])
                acc = metrics.accuracy_score(y_pred,y_n[dev])
                if(acc > best_acc + ACC_THRESH):
                    best_acc = acc
                    best_c = c
    
            # retrain with all train data and best_c
            print('fold:',i,' best c:',best_c, ' dev:%.2f' % best_acc, ' dev_ones:%.2f' % (y_n[dev].sum()/len(dev)),end='')
            if model == 'LOGISTIC': 
                clf = linear_model.LogisticRegression(penalty=regularizer,C=best_c)
                clf1 = linear_model.LogisticRegression(penalty=regularizer,
                                                                          C=best_c, n_jobs=-1)                
            elif model == 'KNN':           
                clf = KNeighborsClassifier(n_neighbors=best_c, metric='euclidean',weights='uniform')
            else:
                clf = tree.DecisionTreeClassifier(max_leaf_nodes=best_c)
            clf.fit(X_nd[train],y_n[train])
            y_pred = clf.predict(X_nd)
            acc_test_a[i] = metrics.accuracy_score(y_n[test],y_pred[test])
            acc_train_a[i] = metrics.accuracy_score(y_n[train],y_pred[train])

            f1_test_a[i] = metrics.f1_score(y_n[test],y_pred[test])
            f1_train_a[i] = metrics.f1_score(y_n[train],y_pred[train])

            print(' acc_test:%.2f' % acc_test_a[i], ' acc_train:%.2f' % acc_train_a[i], 
                ' f1_train:%.2f' % f1_train_a[i], ' f1_train:%.2f' % f1_train_a[i])
            
            ##------------------------------------------
            ## Plotting Logisitic reg weights
            d= df.drop('y',axis=1)
            clust = list(d.columns.values)
            # getting new fit using all the data
            clf1.fit(X_nd, y_n)
            if weight_plot == 1:
                coeff=clf1.coef_[0]
                weights_dict= dict(zip(clust,coeff))
                wn0= {}
                for k, v in weights_dict.items():
                    if v == 0:
                        wn0[k] = v
                print ('  ',wn0)
                weight_list = [x+y for x,y in zip(coeff,weight_list)] 
                           
        print('Avg test acc:%.3f' % acc_test_a.mean(),'Avg train acc:%.3f' % acc_train_a.mean(),
            'Avg f1 acc:%.3f' % f1_train_a.mean(),'Avg f1 acc:%.3f' % f1_train_a.mean())
        
        if weight_plot == 1:
            avgWeights= [x/N_FOLDS for x in weight_list]
            x= clust
            pos =[x for x in range(0,len(x))]
            plt.bar(pos,avgWeights, align='center', alpha=0.5)
            plt.xticks(pos, x,rotation=90)
        
            plt.ylabel('Avg Weight')
            plt.title('y = '+ y_col+' model '+model )
        
            plt.show()              

#----------------------------------------------------------------------

def plot(f0,f1,df):
    df = df[[f0,f1,'y']]
    df = df.dropna()
    
    plt.scatter(df[f0],df[f1],c=['g' if x else 'r' for x in df['y']],alpha=0.3)
    plt.xlabel(f0)
    plt.ylabel(f1)
    plt.show()

#=============================================================================

def run_models(df):

    y_list = ['Engaged_Jobs','Engaged_Fair','Engaged_Appointment']


    for y in y_list:
        df['y'] = df[y]
        
        feat_names = ['US Citizen','School Year Name','Educations Cumulative Gpa','Documents Count','Engaged_Fair',
                    'Engaged_Appointment','Engaged_Jobs']        

        # feat_names = ['US Citizen','School Year Name','Educations Cumulative Gpa','Documents Count','Drop_in_advisor',
        #             'Appointment Type Length (Minutes)','days_before_due','pre_reg', 'check_in','Engaged_Fair',
        #             'Engaged_Appointment']#, 'Engaged_Jobs']

        print('target variable ', y)
        print('df.shape:',df.shape)

        feat_names.remove(y)
        #feat_names = set(feat_names) - set(y)
        
        usecols = ['y'] + list(feat_names)

        for c in df[usecols].columns:
            print(c,end=',')

        ml(df[usecols],['LOGISTIC'],y_col = y, weight_plot=1)


if __name__ == '__main__':

    print('starting ml')

    datafile = '../data/all_data_numeric.csv'    
    df = pd.read_csv(datafile, skipinitialspace=True) 

    run_models(df)
    
    
    print('end')