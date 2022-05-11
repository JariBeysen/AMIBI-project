import synapseclient 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

syn = synapseclient.Synapse() 
# Login to synapse account
syn.login('name','password') 
 
A_normal = syn.get('syn1445997')#LUSC
A_tumor = syn.get('syn415758')
B_normal = syn.get('syn1446012')#READ
B_tumor = syn.get('syn416194')
C_tumor = syn.get('syn412284')#GBM
D_normal = syn.get('syn1446023')#UCEC
D_tumor = syn.get('syn416204')
E_normal = syn.get('syn1445965')#COAD
E_tumor = syn.get('syn411993')
F_normal = syn.get('syn1446006')#OV
F_tumor = syn.get('syn415945')
G_tumor = syn.get('syn1571536')#LAML
H_normal = syn.get('syn1571460')#LUAD
H_tumor = syn.get('syn1571458')
I_normal = syn.get('syn1445954')#BRCA
I_tumor = syn.get('syn411485')
J_normal = syn.get('syn1445982')#KIRC
J_tumor = syn.get('syn412701')

An,At,Bn,Bt,Ct,Dn,Dt,En,Et,Fn,Ft,Gt,Hn,Ht,In,It,Jn,Jt = [A_normal.path,A_tumor.path,B_normal.path,B_tumor.path,C_tumor.path
,D_normal.path,D_tumor.path,E_normal.path,E_tumor.path,F_normal.path,F_tumor.path,G_tumor.path,H_normal.path,H_tumor.path
,I_normal.path,I_tumor.path,J_normal.path,J_tumor.path]
n_feat = 1000000


    

An_rows = []
with open(An ) as f:
    header = f.readline().split('\t')
    for i,line in enumerate(f):
        col = [x.strip() for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        An_rows.append(col)
        
At_rows = []
with open(At) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        At_rows.append(col)
        
Bn_rows = []
with open(Bn) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Bn_rows.append(col)

        
Bt_rows = []
with open(Bt) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Bt_rows.append(col)
        
Ct_rows = []
with open(Ct) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Ct_rows.append(col)
        
Dn_rows = []
with open(Dn) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Dn_rows.append(col)

        
Dt_rows = []
with open(Dt) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Dt_rows.append(col)
        
En_rows = []
with open(En) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        En_rows.append(col)

        
Et_rows = []
with open(Et) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Et_rows.append(col)
        
Fn_rows = []
with open(Fn) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Fn_rows.append(col)

        
Ft_rows = []
with open(Ft) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Ft_rows.append(col)
        
Gt_rows = []
with open(Gt) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Gt_rows.append(col)
        
Hn_rows = []
with open(Hn) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Hn_rows.append(col)

        
Ht_rows = []
with open(Ht) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Ht_rows.append(col)
        
In_rows = []
with open(In) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        In_rows.append(col)

        
It_rows = []
with open(It) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        It_rows.append(col)
        
Jn_rows = []
with open(Jn) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Jn_rows.append(col)

        
Jt_rows = []
with open(Jt) as f:
    header = f.readline()
    for i,line in enumerate(f):
        col = [x.strip()  for x in line.split('\t')][1:]
        x_less_list = [float(x) for x in col if x!="NA"]
        xm = np.mean(x_less_list)
        col = [float(x) if x!= "NA" else xm for x in col]
        col = [ x if 0<=x<=1 else 0 for x in col]
        if i>=n_feat:
            break
        Jt_rows.append(col)



data = [np.transpose(An_rows),np.transpose(At_rows),np.transpose(Bn_rows),np.transpose(Bt_rows),
np.transpose(Ct_rows),np.transpose(Dn_rows),np.transpose(Dt_rows),np.transpose(En_rows),np.transpose(Et_rows),
np.transpose(Fn_rows),np.transpose(Ft_rows),np.transpose(Gt_rows),np.transpose(Hn_rows),np.transpose(Ht_rows),
np.transpose(In_rows),np.transpose(It_rows),np.transpose(Jn_rows),np.transpose(Jt_rows),]


labellist = [[1,0],[0,1],[1,0],[0,1],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]]

max_var_inds = np.arange(0,len(data[0][0]))


dna= []
labels = []


for i,d in enumerate(data):
    
    label = labellist[i]
    
    for k in d:
        sub = []
        for ind in range(len(k)):
            sub.append(k[ind])
            
        dna.append(sub)
        
        labels.append(label)
      
rand_ind = list(np.arange(0,len(dna)))
random.shuffle(rand_ind)
dna_shuf = []
labels_shuf = []

for ind in rand_ind:
    dna_shuf.append(dna[ind])
    labels_shuf.append(labels[ind])
    
full_data = [dna_shuf,labels_shuf]
 

file1 = open("DNA.txt",'wb')
pickle.dump(full_data,file1)    
file1.close()    
    
    