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
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'run_im/')

start_time = time.monotonic()
callbacks = False
class V_autoencoder(tf.keras.Model):
    
    def __init__(self,lat_size,in_out_dim,label_dim,layers,activation='relu',kernel_initializer='he_normal',):
        
        super().__init__()
        self.num_hidden_layers = layers
        
        self.in_out_dim = in_out_dim
        self.label_dim = label_dim
        self.lat_size = lat_size
        self.latent_mean = tf.keras.layers.Dense(lat_size)
        self.latent_sigma = tf.keras.layers.Dense(lat_size)
        
        self.e_hidden = [tf.keras.layers.Dense(int(self.in_out_dim*.1-(self.in_out_dim*.1-self.lat_size)/(self.num_hidden_layers+1)*(k+1)),
                             activation=activation,
                             kernel_initializer=kernel_initializer,use_bias=False)
                           for k in range(self.num_hidden_layers)]

        self.d_hidden = [tf.keras.layers.Dense(int(self.lat_size+(self.in_out_dim*.1-lat_size)/(self.num_hidden_layers+1)*(k+1)),
                             activation=activation,
                             kernel_initializer=kernel_initializer,use_bias=False)
                           for k in range(self.num_hidden_layers)]
        
        self.i_hidden = [tf.keras.layers.Dense(self.lat_size,
                             activation=activation,
                             kernel_initializer=kernel_initializer,use_bias=False)
                           for k in range(1)]
        
        self.i_out = tf.keras.layers.Dense(self.label_dim,activation="sigmoid",use_bias=True)
        
        self.out = tf.keras.layers.Dense(self.in_out_dim,activation=activation)
        
        
        
        
                           
    def encoder(self,X):
        
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.e_hidden[i](Z)
            Z = tf.keras.layers.Dropout(0.05)(Z)

        out_mean = self.latent_mean(Z)
        out_sigma = self.latent_sigma(Z)
        return out_mean,out_sigma
    
    def decoder(self,X):
        Z = X
        for i in range(self.num_hidden_layers):
            Z = self.d_hidden[i](Z)
            Z = tf.keras.layers.Dropout(0.05)(Z)
            
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
        
        kl_loss = -0.1*tf.reduce_mean(1 + z_sigma - tf.square(z_mean) - tf.exp(z_sigma))
        
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
                
                if lab == 0:
                    plt.scatter(l[0],l[1],color = 'blue')
                if lab == 1:
                    plt.scatter(l[0],l[1],color = 'red')
                    
            plt.scatter([],[],color = 'blue',label = 'normal')
            plt.scatter([],[],color = 'red',label = 'tumor')
            plt.title("train accuracy = "+str(acc))
                    
            plt.legend()
            
            

            fig.savefig(results_dir+"latent space"+str(self.iter))
            
            plt.close(fig)  
        
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
    
    def full_latent_space_clas(self):
        
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

            lab = list(l).index(max(list(l)))           
            if lab == 0:
                plt.scatter(i[0],i[1],color = "blue")
        
            if lab == 1:
                plt.scatter(i[0],i[1],color = "red")
        
        
        
        
        

def labeled_dna(label,l,d):
    
    ld = [y for x,y in zip(l,d) if list(x) == label]
        
    ll = [x for x,y in zip(l,d) if list(x) == label]
        
    return np.array(ll),np.array(ld)

def p_fn(t,n,tc,nc):
    
    return ((1-tc)*t)/(((1-tc)*t) + nc*n)
        

    
def VAE(data,test_data):
    

        
    dnas,labels = data
    
    test_dnas,test_labels = test_data
    
    
    
    tumor_labels,tumor_dnas = labeled_dna([0,1], labels, dnas)

    
    normal_labels,normal_dnas = labeled_dna([1,0], labels, dnas)
    
    Tum_len = len(tumor_labels)
    
    n_samples = len(dnas)+ len(test_dnas)
    
    train_len = len(dnas)
        
    print("percentage of data is tumor: "+str(100*Tum_len/train_len)+"%")
    
    test_tumor_labels,test_tumor_dnas = labeled_dna([0,1], test_labels, test_dnas)
    
    test_normal_labels,test_normal_dnas = labeled_dna([1,0], test_labels, test_dnas)
    
    test_tumor_labels = np.array(test_tumor_labels,dtype='float32')
    
    test_tumor_dnas = np.array(test_tumor_dnas,dtype='float32')
    
    test_normal_labels = np.array(test_normal_labels,dtype='float32')
    
    test_normal_dnas = np.array(test_normal_dnas,dtype='float32')
    
    tf_test_tumor_dnas = tf.data.Dataset.from_tensor_slices(test_tumor_dnas).batch(1)
    
    tf_test_tumor_labels = tf.data.Dataset.from_tensor_slices(test_tumor_labels).batch(1)
    
    tf_test_normal_dnas = tf.data.Dataset.from_tensor_slices(test_normal_dnas).batch(1)
    
    tf_test_normal_labels = tf.data.Dataset.from_tensor_slices(test_normal_labels).batch(1)

    
    tumor_labels = np.array(tumor_labels,dtype='float32')
    
    tumor_dnas = np.array(tumor_dnas,dtype='float32')
    
    normal_labels = np.array(normal_labels,dtype='float32')
    
    normal_dnas = np.array(normal_dnas,dtype='float32')

    tf_labels = tf.data.Dataset.from_tensor_slices(labels).prefetch(30)
    
    tf_dna = tf.data.Dataset.from_tensor_slices(dnas).prefetch(30)
    
    tf_test_labels = tf.data.Dataset.from_tensor_slices(test_labels).batch(1).prefetch(30)
    
    tf_test_dnas = tf.data.Dataset.from_tensor_slices(test_dnas).batch(1).prefetch(30)
    
    tf_tumor_dnas = tf.data.Dataset.from_tensor_slices(tumor_dnas).batch(1)
    
    tf_tumor_labels = tf.data.Dataset.from_tensor_slices(tumor_labels).batch(1)
    
    tf_normal_dnas = tf.data.Dataset.from_tensor_slices(normal_dnas).batch(1)
    
    tf_normal_labels = tf.data.Dataset.from_tensor_slices(normal_labels).batch(1)


    n_feat = len(dnas[0])
    n_lab = len(labels[0])
    n_samples = len(dnas)
    print( 'sample # = '+str(n_samples))
    
    model = V_autoencoder(10, n_feat,n_lab, 3)
    
    optimizer = tf.keras.optimizers.Nadam()
    
    lr = 2e-6
    
    solv = VAE_Solver(model,[tf_dna,tf_labels])
    
    solv.train(optimizer, lr,50,30)
        
    
    test_ac = solv.eval(tf_test_dnas,tf_test_labels)
    print("Accuracy = " +str(test_ac))
    
    test_ac_t = solv.eval(tf_test_tumor_dnas,tf_test_tumor_labels)
    
    test_ac_n = solv.eval(tf_test_normal_dnas,tf_test_normal_labels)
    
    gen_labels,gen_dnas = solv.generate_dna([0,1],tf_tumor_dnas, 1000)
    
    tf_gen_labels = tf.data.Dataset.from_tensor_slices(gen_labels).batch(1)
    
    tf_gen_dnas = tf.data.Dataset.from_tensor_slices(gen_dnas).batch(1)
    
    test_ac_gt = solv.eval(tf_gen_dnas,tf_gen_labels)
    print("tumor gen accuracy = " +str(test_ac_gt))
    
 
    gen_labels,gen_dnas = solv.generate_dna([1,0],tf_normal_dnas, 1000)
    
    tf_gen_labels = tf.data.Dataset.from_tensor_slices(gen_labels).batch(1)
    
    tf_gen_dnas = tf.data.Dataset.from_tensor_slices(gen_dnas).batch(1)
    
    test_ac_gn = solv.eval(tf_gen_dnas,tf_gen_labels)
    print("normal gen accuracy = " +str(test_ac_gn))
    
    
    
    latent_space,latent_labels = solv.latent_space(tf_gen_dnas,tf_gen_labels)
    pca = PCA(2)
    pca.fit(latent_space)
    latent_space = pca.transform(latent_space)
    
    for ind,l in enumerate(latent_space):
        
        lab = list(latent_labels[ind]).index(max(list(latent_labels[ind])))
        
        if lab == 0:
            plt.scatter(l[0],l[1],color = 'blue')
        if lab == 1:
            plt.scatter(l[0],l[1],color = 'red')
            
    fn = p_fn(Tum_len/train_len,1-Tum_len/train_len,test_ac_t,test_ac_n)
            
    print("false negative percentage = "+str(fn*100)+"%")
            
    
    return [test_ac,test_ac_n,test_ac_t,fn,test_ac_gn,test_ac_gt]

def stat(n):
    
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
        
    print(len(X_train))

    vals = []
    for ind in range(n):
        
        vals.append(VAE([X_train[ind],Y_train[ind]],[X_test[ind],Y_test[ind]]))
    

    return np.mean(vals,axis=0),np.std(vals,axis=0)


print(stat(2))

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
winsound.Beep(1000,100)
