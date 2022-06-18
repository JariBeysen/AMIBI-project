# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:59:03 2022

@author: jarim
"""
import os
import random
import time
import warnings
from datetime import timedelta
from math import *
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pickle
import winsound
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import tensorflow
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'run_im/')

start_time = time.monotonic()
callbacks = True

class V_autoencoder(tf.keras.Model):
    
    def __init__(self,lat_size,in_out_dim,label_dim,layers,activation='gelu',kernel_initializer='he_normal',):
        
        super().__init__()
        self.num_hidden_layers = layers
        
        self.in_out_dim = in_out_dim
        self.label_dim = label_dim
        self.lat_size = lat_size
        self.latent_mean = tf.keras.layers.Dense(lat_size)
        self.latent_sigma = tf.keras.layers.Dense(lat_size)
        activation = self.act_func
        scaler=0.01

        
        self.e_hidden = [tf.keras.layers.Dense(int(self.in_out_dim*scaler-(self.in_out_dim*scaler-self.lat_size)/(self.num_hidden_layers+1)*(k+1)),
                             activation=activation,
                             kernel_initializer=kernel_initializer,use_bias=False)
                           for k in range(self.num_hidden_layers)]

        self.d_hidden = [tf.keras.layers.Dense(int(self.lat_size+(self.in_out_dim*scaler-lat_size)/(self.num_hidden_layers+1)*(k+1)),
                             activation=activation,
                             kernel_initializer=kernel_initializer,use_bias=False)
                           for k in range(self.num_hidden_layers)]
        
        self.i_hidden = [tf.keras.layers.Dense(self.lat_size,
                             activation='relu',
                             kernel_initializer=kernel_initializer,use_bias=False)
                           for k in range(1)]
        
        self.i_out = tf.keras.layers.Dense(self.label_dim,activation="sigmoid",use_bias=True)
        
        self.out = tf.keras.layers.Dense(self.in_out_dim,activation=activation)
        
        
        
    def act_func(self,x):
        return tf.math.erf(x)
                           
    def encoder(self,X):
        
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.e_hidden[i](Z)
            Z = tf.keras.layers.AlphaDropout(0.01)(Z)

        out_mean = self.latent_mean(Z)
        out_sigma = self.latent_sigma(Z)
        return out_mean,out_sigma
    
    def decoder(self,X):
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.d_hidden[i](Z)
            Z = tf.keras.layers.AlphaDropout(0.05)(Z)
            
        out = self.out(Z)

        return out
    
    def identifier(self,X):
        Z = X
        for i in range(1):
            Z = self.i_hidden[i](Z)
        
        return self.i_out(Z)
        
        
    def sampling(self, X):
        z_mean, z_log_var = X
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def vae(self,X):
        
        e = self.encoder(X)
        l = self.sampling(e)
        d = self.decoder(l)

        return d
    
    def call(self,X):
        
        Y = self.vae(X)
        
        return Y

class VAE_Solver():
    
    def __init__(self, model,data):
        self.model = model
        self.iter = 0
        self.dnas = data[0]
        self.labels = data[1]
        
    def loss_fn(self,x,y):
        
        x_pred = self.model(x)
         
        mse_loss = tf.reduce_mean(tf.square(x-x_pred))
        
        z_mean,z_sigma = self.model.encoder(x)
        
        kl_loss = -0.001*tf.reduce_mean(1 + z_sigma - tf.square(z_mean) - tf.exp(z_sigma))
        
        sampl = self.model.sampling([z_mean,z_sigma])
        
        y_pred = self.model.identifier(sampl)
        
        ce = tf.keras.losses.CategoricalCrossentropy()(tf.reshape(y,(y.shape[0],y.shape[1])), tf.reshape(y_pred,(y.shape[0],y.shape[1])))
        
        loss = kl_loss+ mse_loss +ce
        
        return loss,mse_loss,kl_loss,ce
    
    
    def get_grad(self,x,y):
      
        with tf.GradientTape(persistent=True) as tape:
                
            tape.watch(self.model.trainable_variables)
            loss,mse_loss,kl_loss,ce = self.loss_fn(x,y)
                
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
            
        return loss,mse_loss,kl_loss,ce,g
    
        
    def train(self,optimizer,lr,batchsize, N=100):
        optimizer.learning_rate.assign(lr)
        dnas = self.dnas
        labels = self.labels
        
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        labellist = [ np.random.uniform(0,1,3).tolist() for j in range(18)]
        def colorind(x):
            lab = list(x).index(max(list(x)))    
            return labellist[lab]
        def callback(acc):
            
            X = dnas.batch(1)
            Y = labels.batch(1)
            
            latent_space,latent_labels = self.latent_space(X,Y)
            
            pca = PCA(2)
            pca.fit(latent_space)
            latent_space = pca.transform(latent_space)
            
            fig = plt.figure(figsize=(10,10))
            
            for ind,l in enumerate(latent_space):
                
                lab = list(latent_labels[ind]).index(max(list(latent_labels[ind])))
                
                plt.scatter(l[0],l[1],color = colorind(latent_labels[ind]) )
                    

            plt.title("train accuracy = "+str(acc))
                    
            
            

            fig.savefig(results_dir+"latent space"+str(self.iter))
            
            plt.close(fig)  
        def lr_red(i, N):
            optimizer.learning_rate.assign(tf.math.abs(tf.random.normal((),lr*(1-i/N*.9),1/5*lr*(1-i/N*.9))))
            
        def train_step():

            rand_seed = np.random.randint(0,10000)
            
            X = dnas.shuffle(buffer_size=1,seed=rand_seed).batch(batchsize)
            Y = labels.shuffle(buffer_size=1,seed=rand_seed).batch(batchsize)

            for x,y in zip(X,Y):

                loss,mse_loss,kl_loss,ce,grad = self.get_grad(x,y)
                
                variables = self.model.trainable_variables
    
                optimizer.apply_gradients(zip(grad, variables))
                
                enc = self.model.encoder(x)
                sampl = self.model.sampling(enc)
                y_pred = self.model.identifier(sampl)

                train_acc_metric.update_state(tf.reshape(y,(y.shape[0],y.shape[1])), tf.reshape(y_pred,(y.shape[0],y.shape[1])))
                
            
            train_acc = train_acc_metric.result()
            
            train_acc_metric.reset_states()
            if callbacks:
                callback(train_acc.numpy())
            return loss,mse_loss,kl_loss,ce,train_acc
        
         
   
        for i in range(N):

            loss,mse_loss,kl_loss,ce,train_acc = train_step()
            lr_red(i,N)
            # if train_acc==1:
            #     break

            if i%1 ==0:
                
                print("Epoch/loss/mse/kl/ce")
                tf.print(i,loss,mse_loss,kl_loss,ce)
                print("Accuracy = " +str(train_acc.numpy()))
                print("_______________________________")
                
            self.iter+=1
            
    def predict(self,X):
        
        
        Z = self.model.encoder(X)
        
        sampl = self.model.sampling(Z)

        i = self.model.identifier(sampl)

        return i
    
    def eval(self,X,Y):
        test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        for x,y in zip(X,Y):
            enc = self.model.encoder(x)
            sampl = self.model.sampling(enc)
            y_pred = self.model.identifier(sampl)

            test_acc_metric.update_state(tf.reshape(y,(y.shape[0],y.shape[1])), tf.reshape(y_pred,(y.shape[0],y.shape[1])))
        test_acc = test_acc_metric.result()
        return test_acc.numpy()
    def latent_space(self,X,Y):
        
        vals= []
        labels = []

        
        for x,y in zip(X,Y):
            
            
            x = self.model.encoder(x)
            sampl = self.model.sampling(x)
            label = self.model.identifier(sampl)
            
            labels.append(label[0].numpy())
            vals.append(x[0][0].numpy())
            
        return vals,labels
    
    def generate_dna(self,label,dna,n):
        
        means = []
        sigmas = []
        for x in dna:
            mean,sigma = self.model.encoder(x)
            
            means.append(mean)
            sigmas.append(sigma)
        
        avg_enc = tf.reduce_mean(means,axis=0)
        
        std_enc = tf.reduce_mean(sigmas,axis=0)
        
        gen_dnas = []
        
        gen_labels = []
        
        for i in range(n):
            
            sampl = self.model.sampling([avg_enc,std_enc])
            
            gen_dna = self.model.decoder(sampl)
            
            gen_dnas.append(gen_dna.numpy()[0])
            
            gen_labels.append(label)
            
        return np.array(gen_labels),np.array(gen_dnas)
    
    def generate_dna(self,label,dna,n):
        try:
            means = []
            sigmas = []
            for x in dna:
                mean,sigma = self.model.encoder(x)
                
                means.append(mean)
                sigmas.append(sigma)
            
            max_enc = tf.reduce_max(means,axis=0)
            
            min_enc = tf.reduce_min(means,axis=0)
            
            max_enc = tfp.stats.percentile(means,90,axis=0)
            
            min_enc = tfp.stats.percentile(means,10,axis=0)
            
            gen_dnas = []
            
            gen_labels = []
            
            for i in range(n):
                
                sample = []
                
                for x,y in zip(min_enc,max_enc):
                    
                    sample.append(tf.random.uniform((1,1),x,y))
                    
                sample = tf.concat(sample, axis=1)
                
                gen_dna = self.model.decoder(sample)
                
                gen_dnas.append(gen_dna.numpy()[0])
                
                gen_labels.append(label)
                
            return np.array(gen_labels),np.array(gen_dnas)
        
        except:
            return np.zeros((1,len(label))),np.zeros((1,dna.shape[1],dna.shape[2]))
    
    def full_latent_space_clas(self):
        labellist = [ np.random.uniform(0,255,3).tolist() for j in range(18)]
        def colorind(x):
            lab = list(x).index(max(list(x)))    
            return labellist[lab]
        
        
        
        X = [np.random.uniform(-2,2,size=(1,self.model.lat_size)) for i in range(10000)]
        
        space = []
        
        labels = []
        
        for x in X:
             
             v = tf.constant(x,dtype="float32")
                
             ID = self.model.identifier(v)
                
             space.append(x[0])
             
             labels.append(ID.numpy()[0])
             
        pca = PCA(2)
        pca.fit(space)
        space = pca.transform(space)
                
        plt.figure()        
        
        for i,l in zip(space,labels):

            
            plt.scatter(i[0],i[1],color = colorind(l) )
        
           
                

def labeled_dna(label,l,d):
    
    ld = [y for x,y in zip(l,d) if list(x) == label]
        
    ll = [x for x,y in zip(l,d) if list(x) == label]
        
    return ll,ld


def data_label_tf(labellist,labels,dnas):
    
    data_dna = []
    data_label = []
    
    for x in labellist:
        
        label,dna = labeled_dna(x,labels,dnas)
        
        dna_tf = tf.data.Dataset.from_tensor_slices(np.array(dna)).batch(1)
        
        label_tf = tf.data.Dataset.from_tensor_slices(np.array(label)).batch(1)
        
        data_dna.append(dna_tf)
        
        data_label.append(label_tf)
        
    return data_label,data_dna

def data_label(labellist,labels,dnas):
    
    data_dna = []
    data_label = []
    
    for x in labellist:
        
        label,dna = labeled_dna(x,labels,dnas)
        
        
        data_dna.append(dna)
        
        data_label.append(label)
        
    return np.array(data_label),np.array(data_dna)
    
    
def VAE(data,test_data):
    

        
    dnas,labels = data
    
    test_dnas,test_labels = test_data
    
    
    n_samples = len(dnas)+ len(test_dnas)
    
    train_len = len(dnas)

    tf_labels = tf.data.Dataset.from_tensor_slices(labels)
    
    tf_dna = tf.data.Dataset.from_tensor_slices(dnas)
    
    tf_test_labels = tf.data.Dataset.from_tensor_slices(test_labels).batch(1)
    
    tf_test_dnas = tf.data.Dataset.from_tensor_slices(test_dnas).batch(1)
    



    n_feat = len(dnas[0])
    n_lab = len(labels[0])
    n_samples = len(dnas)
    print( 'sample # = '+str(n_samples))
    
    model = V_autoencoder(30, n_feat,n_lab, 3)
    
    optimizer = tf.keras.optimizers.Nadam()
    
    lr = 2e-4
    
    solv = VAE_Solver(model,[tf_dna,tf_labels])
    
    solv.train(optimizer, lr,20,20)
        
    
    test_ac = solv.eval(tf_test_dnas,tf_test_labels)
    print("Accuracy = " +str(test_ac))
    
    labellist = [[1 if j ==k else 0 for k in range(18)] for j in range(18)]
    
    sorted_labels,sorted_dna = data_label_tf(labellist, test_labels, test_dnas)
    
    test_acs = [solv.eval(x,y) for y,x in zip(sorted_labels,sorted_dna)]

    print("Individual accuracy = " +str(test_acs))
    
    
    # gen_data = [solv.generate_dna(y,x, 100) for y,x in zip(labellist,sorted_dna)]
    
    # gen_data_tf = [ (tf.data.Dataset.from_tensor_slices(x[0]).batch(1), tf.data.Dataset.from_tensor_slices(x[1]).batch(1)) for x in gen_data ]

    
    # test_ac_gen = [solv.eval(x[1],x[0]) for x in gen_data_tf]
    
    
    # print("Gen idnividual accuracy = " +str(test_ac_gen))

    
    return [test_ac,test_acs]#,test_ac_gen


def oneD_class(x):

    lab = list(x).index(max(list(x)))

    return lab

def twoD_class(x,l):

    lab = [1 if i == x else 0 for i in range(l)]

    return lab
    

    
def stat(n):
    
    file1 = open("DNA.txt",'rb')

    data = pickle.load(file1)
    
    file1.close()
    
    dna,labels = data
    
    labellist = [[1 if j ==k else 0 for k in range(18)] for j in range(18)]
    
    npsl,_ = data_label(labellist, labels, dna)
    
    for s in npsl:
        print(len(s))
    
    labels = [oneD_class(x) for x in labels]
    
    dna = np.array(dna,dtype="float32")
    
    labels = np.array(labels,dtype="float32")
    
    split = StratifiedKFold(n,shuffle=True)

    X_train,Y_train = [],[]
    X_test,Y_test = [],[]


    for train,test in split.split(dna,labels):
        
        X_train.append(dna[train])
        Y_train.append(labels[train])
        X_test.append(dna[test])
        Y_test.append(labels[test])
        
    Y_train = np.array([[twoD_class(x, len(labellist[0])) for x in y] for y in Y_train])
    Y_test = np.array([[twoD_class(x, len(labellist[0])) for x in y] for y in Y_test])
    
    vals_f = []
    vals_ind = []
    for ind in range(n):
        
        run = VAE([X_train[ind],Y_train[ind]],[X_test[ind],Y_test[ind]])
        vals_ind.append(run[0])
        vals_f.append(run[1])
    

    return np.mean(vals_f,axis=0),np.std(vals_f,axis=0),np.mean(vals_ind,axis=0),np.std(vals_ind,axis=0)
    

def single(n):
    
    file1 = open("DNA.txt",'rb')

    data = pickle.load(file1)
    
    file1.close()
    
    dna,labels = data
    
    dna = np.array(dna,dtype="float32")
    
    labels = np.array(labels,dtype="float32")
    
    split = KFold(n,shuffle=True)

    X_train,Y_train = [],[]
    X_test,Y_test = [],[]


    for train,test in split.split(dna,labels):
        
        X_train.append(dna[train])
        Y_train.append(labels[train])
        X_test.append(dna[test])
        Y_test.append(labels[test])
        

    vals_f = []
    vals_ind = []
    for ind in range(1):
        run = VAE([X_train[ind],Y_train[ind]],[X_test[ind],Y_test[ind]])
        vals_ind.append(run[0])
        vals_f.append(run[1])
    

    return np.mean(vals_f,axis=0),np.std(vals_f,axis=0),np.mean(vals_ind,axis=0),np.std(vals_ind,axis=0)


print(stat(10))

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
winsound.Beep(1000,100)