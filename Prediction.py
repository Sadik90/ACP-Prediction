import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from readFasta import readFasta
from AAINDEX import AAINDEX
from AAC import AAC
import re


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


def prediction(fasta_seq):
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
    model= pickle.load(open('model.pkl','rb'))
    score=model.predict(featurization)
    return (score>=0.5)*1

print(prediction('RRWQWR'))
