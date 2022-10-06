import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from readFasta import readFasta
from AAINDEX import AAINDEX
from AAC import AAC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm


import seaborn as sns
import matplotlib.pyplot as plt



# Load libraries

from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def calculate_performace(test_num,y_pred_xgb,val_label):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if val_label[index] == 1:
            if val_label[index] == y_pred_xgb[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if val_label[index] == y_pred_xgb[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, precision, sensitivity, specificity, MCC

def BPF(seq_temp):
    seq = seq_temp
    # chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    fea = []
    tem_vec =[]
    k = 7
    for i in range(k):
        if seq[i] =='A':
            tem_vec = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='C':
            tem_vec = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='D':
            tem_vec = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='E':
            tem_vec = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='F':
            tem_vec = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='G':
            tem_vec = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='H':
            tem_vec = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='I':
            tem_vec = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='K':
            tem_vec = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='L':
            tem_vec = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='M':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='N':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif seq[i]=='P':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif seq[i]=='Q':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif seq[i]=='R':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif seq[i]=='S':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif seq[i]=='T':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif seq[i]=='V':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif seq[i]=='W':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif seq[i]=='Y':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        fea = fea + tem_vec
    return fea

def get_features(fasta_seq):
    features = ['SeqPos.1.NADH010107', 'SeqPos.8.AURR980105', 'SeqPos.24.RACS820102', 'SeqPos.4.PALJ810114',
                'SeqPos.44.CHOP780201', 'SeqPos.3.ARGP820102', 'SeqPos.35.TANS770104', 'SeqPos.8.GEOR030107',
                'SeqPos.41.RICJ880117', 'SeqPos.25.GEOR030104', 'SeqPos.1.NAKH900110', 'SeqPos.48.RACS770103',
                'SeqPos.2.PONP800108', 'SeqPos.8.AURR980114', 'SeqPos.1.YUTK870104', 'SeqPos.43.GEIM800110',
                'SeqPos.4.MEIH800102', 'SeqPos.1.AURR980107', 'SeqPos.24.GEOR030105', 'SeqPos.44.RICJ880112',
                'SeqPos.14.FAUJ880113', 'SeqPos.50.TANS770104', 'SeqPos.7.ROBB760102', 'SeqPos.1.GARJ730101',
                'SeqPos.4.CHAM830101', 'SeqPos.44.CHAM830104', 'SeqPos.8.CIDH920103', 'SeqPos.41.TANS770102',
                'SeqPos.1.WERD780103', 'SeqPos.1.FINA910101', 'SeqPos.8.FINA770101', 'SeqPos.43.OOBM850102',
                'SeqPos.1.LEVM780101', 'SeqPos.18.BUNA790103', 'SeqPos.50.YUTK870104', 'SeqPos.1.PTIO830102',
                'SeqPos.4.FAUJ880110', 'SeqPos.8.RICJ880111', 'SeqPos.41.PONP930101', 'SeqPos.2.RICJ880111',
                'SeqPos.1.GEOR030107', 'SeqPos.3.GUYH850101', 'SeqPos.44.GEIM800103', 'SeqPos.16.OOBM850102',
                'SeqPos.1.BLAS910101', 'SeqPos.48.HOPA770101', 'SeqPos.4.GEIM800108', 'SeqPos.1.YUTK870101',
                'SeqPos.44.AURR980114', 'SeqPos.14.JOND750102']
    # fasta1 = readFasta(fix_fasta_file_path)
    if len(fasta_seq)<50:
        fasta_seq_50=fasta_seq+'-'*(50-len(fasta_seq))
    else:
        fasta_seq_50=fasta_seq[:50]
    aaindex = AAINDEX([['1',fasta_seq_50]])
    import pandas as pd
    data = pd.DataFrame(aaindex[1:], columns=[''] + aaindex[0])
    data = data[features]
    AAindex = data.to_numpy()
    # fasta2 = readFasta(fasta_file)
    aac = AAC([['1',fasta_seq]], order=None)
    bp = np.array([BPF(fasta_seq_50)])
    featurization = np.concatenate([bp, AAindex, aac], axis=1)
    return featurization



label=[]
X_data=[]
pssm=[]
with open('acp740_50.fasta') as f:
    for line in f.readlines():
        if line[0]=='>':
            label.append(int(line.split('|')[1]))
            header='pssm/PSSM740/train_740/pssm/'+line.split('|')[0][1:]+'.npy'
            a=np.load(header)
            a=(a-np.min(a))/(np.max(a)-np.min(a))
            a=a.flatten()
            pssm.append(a)
        else:
            line=line.replace('X','-')
            X_data.append(get_features(line))
X_pssm=np.vstack(pssm)
X_dat=np.concatenate(X_data,axis=0)
X_data=np.hstack([X_dat,X_pssm])
#
val_label=[]
val_X_data=[]
val_pssm=[]
with open('acp240_50.fasta') as f:
    for line in f.readlines():
        if line[0]=='>':
            val_label.append(int(line.split('|')[1]))
            header='pssm/PSSM240/pssm/'+line.split('|')[0][1:]+'.npy'
            a=np.load(header)
            a=(a-np.min(a))/(np.max(a)-np.min(a))
            a=a.flatten()
            val_pssm.append(a)
        else:
            line=line.replace('X','-')
            val_X_data.append(get_features(line))
val_X_pssm=np.vstack(val_pssm)
val_X_dat=np.concatenate(val_X_data,axis=0)
val_X_data=np.hstack([val_X_dat,val_X_pssm])


# #Model for Prediction

clf1 = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100),alpha = 0.1,activation = 'tanh', random_state=120,max_iter=500,solver='adam',learning_rate='adaptive')
clf2 = RandomForestClassifier(random_state=120,n_estimators=300,min_samples_split=10,min_samples_leaf=1,max_features='auto',bootstrap=False)
clf3 = ExtraTreesClassifier(random_state=120,n_estimators=400,max_depth=32,min_samples_split=10,min_samples_leaf=1,max_features='auto',bootstrap=False)
clf4 = KNeighborsClassifier(n_neighbors =3)
clf5 = svm.SVC(kernel="linear")



# #Boosting Classifier

base_classifier = RandomForestClassifier(random_state=120,n_estimators=300,min_samples_split=10,min_samples_leaf=1,max_features='auto',bootstrap=False)
clf6 = AdaBoostClassifier(base_estimator=base_classifier,random_state = 121,n_estimators=406,learning_rate=0.04)
clf7 = GradientBoostingClassifier(random_state=121, n_estimators=350, learning_rate=0.1, loss='deviance')

classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]

from sklearn.metrics import accuracy_score, log_loss

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
for clf in classifiers:
    clf.fit(X_data, label)
    name = clf.__class__.__name__

    print(name)

#     print('****Results****')
    y_pred_xgb = clf.predict(val_X_data)
    acc = accuracy_score(val_label,y_pred_xgb )
    print("Accuracy: {:.4%}".format(acc))
    print(classification_report(val_label, y_pred_xgb))
    print(confusion_matrix(val_label, y_pred_xgb))
    print(accuracy_score(y_pred_xgb,val_label))
    print(matthews_corrcoef(val_label,y_pred_xgb))

    log_entry = pd.DataFrame([[name, acc * 100, 11]], columns=log_cols)
    log = log.append(log_entry)

    print("=" * 30)

    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()
# '''
# #clf = RandomForestClassifier(max_depth=2, random_state=0)
# #clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
# #clf = KNeighborsClassifier(n_neighbors=3)
# all_performance_lstm = []
# #clf = AdaBoostClassifier(n_estimators=100, random_state=0)
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# model = clf.fit(X_data, label)
# y_pred_xgb = model.predict(val_X_data)
# acc, precision, sensitivity, specificity, MCC = calculate_performace(len(val_label), y_pred_xgb, val_label)
# print(acc, precision, sensitivity, specificity, MCC)
# all_performance_lstm.append([acc, precision, sensitivity, specificity, MCC])
# print('mean performance of Classifiers')
# # Summary of the predictions made by the classifier
# #print(classification_report(val_label, y_pred_xgb))
# #print(confusion_matrix(val_label, y_pred_xgb))
# #print(accuracy_score(y_pred_xgb,val_label))
# #print(matthews_corrcoef(val_label,y_pred_xgb)*100)
# '''