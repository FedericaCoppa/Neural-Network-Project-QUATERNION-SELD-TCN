#
# The SELDnet architecture
#
import tensorflow as tf
import keras
from keras.layers import Bidirectional, MaxPooling2D, Input,Conv2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import torch
import numpy as np
from complexnn import *

# Delete if you don't have a GPU
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


####


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,rnn_size, fnn_size, classification_mode, weights, summary):
    # model definition
    
    
    spec_start = Input(shape=(data_in[-2], data_in[-1], data_in[-3]))
    #print("start input:", spec_start)
    spec_cnn = spec_start


    for i, convCnt in enumerate(pool_size):
        #print(spec_cnn)
        spec_cnn = QuaternionConv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3),data_format='channels_last',padding='same')(spec_cnn)
        #print(spec_cnn)
        
        spec_cnn=QuaternionBatchNorm(num_features=spec_cnn.shape[1])(spec_cnn)
        #print(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        #print(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        #print(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
        #print(spec_cnn)
    # print(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    # print(spec_cnn)
    spec_rnn = Reshape((data_in[-2], -1))(spec_cnn)
    print("spec_rnn:\n")
    print(spec_rnn)
     
    




    #### START TCN ###
    
    
    
    
    ##list of dilation factors
    d=[2**0,2**1,2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9]


    # resblock 1
    spec_resblock1=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[0])(spec_rnn)

    spec_resblock1 = QuaternionBatchNorm(num_features=spec_resblock1.shape[1])(spec_resblock1)

    tanh_out1 = Activation('tanh')(spec_resblock1)
    
    sigm_out1 = Activation('sigmoid')(spec_resblock1)

    spec_act1 = keras.layers.Multiply()([tanh_out1, sigm_out1])
       
    spec_drop1 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act1)
    
    skip_output1 = QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop1)

    res_output1 = keras.layers.Add()([spec_rnn, skip_output1])
    
    
   

    #####

    # resblock 2
    spec_resblock2=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[1])(res_output1)

    spec_resblock2 = QuaternionBatchNorm(num_features=spec_resblock2.shape[1])(spec_resblock2)

    tanh_out2 = Activation('tanh')(spec_resblock2)
    
    sigm_out2 = Activation('sigmoid')(spec_resblock2)

    spec_act2 = keras.layers.Multiply()([tanh_out2, sigm_out2])
       
    spec_drop2 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act2)
    
    skip_output2 = QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop2)

    res_output2 = keras.layers.Add()([res_output1, skip_output2])
    
    
    
    

    #####


    #resblock 3
    spec_resblock3=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[2])(res_output2)

    spec_resblock3 = QuaternionBatchNorm(num_features=spec_resblock3.shape[1])(spec_resblock3)

    tanh_out3 = Activation('tanh')(spec_resblock3)
    
    sigm_out3 = Activation('sigmoid')(spec_resblock3)

    spec_act3 = keras.layers.Multiply()([tanh_out3, sigm_out3])
       
    spec_drop3 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act3)
    
    skip_output3 = QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop3)

    res_output3 = keras.layers.Add()([res_output2, skip_output3])
     
    
    
    

    #####

    #resblock 4
    spec_resblock4=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[3])(res_output3)

    spec_resblock4 = QuaternionBatchNorm(num_features=spec_resblock4.shape[1])(spec_resblock4)

    tanh_out4 = Activation('tanh')(spec_resblock4)
    
    sigm_out4 = Activation('sigmoid')(spec_resblock4)

    spec_act4 = keras.layers.Multiply()([tanh_out4, sigm_out4])
       
    spec_drop4 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act4)
    
    skip_output4 = QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop4)

    res_output4 = keras.layers.Add()([res_output3, skip_output4])
    
    
    
    

    #####

     #resblock 5
    spec_resblock5=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[4])(res_output4)

    spec_resblock5 = QuaternionBatchNorm(num_features=spec_resblock5.shape[1])(spec_resblock5)

    tanh_out5 = Activation('tanh')(spec_resblock5)
    
    sigm_out5 = Activation('sigmoid')(spec_resblock5)

    spec_act5 = keras.layers.Multiply()([tanh_out5, sigm_out5])
       
    spec_drop5 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act5)
    
    skip_output5 = QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop5)

    res_output5 = keras.layers.Add()([res_output4, skip_output5])
    
    
    
    

    #####
     
     #resblock 6
    spec_resblock6=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[5])(res_output5)

    spec_resblock6 = QuaternionBatchNorm(num_features=spec_resblock6.shape[1])(spec_resblock6)

    tanh_out6 = Activation('tanh')(spec_resblock6)
    
    sigm_out6 = Activation('sigmoid')(spec_resblock6)

    spec_act6 = keras.layers.Multiply()([tanh_out6, sigm_out6])
       
    spec_drop6 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act6)
    
    skip_output6 = QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop6)

    res_output6 = keras.layers.Add()([res_output5, skip_output6])
    
   
    
   

    #####



    #resblock 7
    spec_resblock7=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[6])(res_output6)

    spec_resblock7 = QuaternionBatchNorm(num_features=spec_resblock7.shape[1])(spec_resblock7)

    tanh_out7 = Activation('tanh')(spec_resblock7)
    
    sigm_out7 = Activation('sigmoid')(spec_resblock7)

    spec_act7 = keras.layers.Multiply()([tanh_out7, sigm_out7])
       
    spec_drop7 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act7)
    
    skip_output7 = QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop7)

    res_output7 = keras.layers.Add()([res_output6, skip_output7])
    
    
    
    

    #####



    #resblock 8
    spec_resblock8=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[7])(res_output7)

    spec_resblock8 = QuaternionBatchNorm(num_features=spec_resblock8.shape[1])(spec_resblock8)

    tanh_out8 = Activation('tanh')(spec_resblock8)
    sigm_out8 = Activation('sigmoid')(spec_resblock8)

    spec_act8 = keras.layers.Multiply()([tanh_out8, sigm_out8])
       
    spec_drop8 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act8)
    
    skip_output8 = QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop8)

    res_output8 = keras.layers.Add()([res_output7, skip_output8])
    
    
    

    #####

    #resblock 9
    spec_resblock9=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[8])(res_output8)

    spec_resblock9 = QuaternionBatchNorm(num_features=spec_resblock9.shape[1])(spec_resblock8)

    tanh_out9 = Activation('tanh')(spec_resblock9)
    sigm_out9 = Activation('sigmoid')(spec_resblock9)

    spec_act9 = keras.layers.Multiply()([tanh_out9, sigm_out9])
       
    spec_drop9 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act9)
    
    skip_output9= QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop9)

    res_output9 = keras.layers.Add()([res_output8, skip_output9])
    
    
    
    

    #####

    #resblock 10
    spec_resblock10=QuaternionConv1D(filters=256,kernel_size=(3),padding='same',dilation_rate=d[9])(res_output9)

    spec_resblock10 = QuaternionBatchNorm(num_features=spec_resblock10.shape[1])(spec_resblock10)

    tanh_out10 = Activation('tanh')(spec_resblock10)
    sigm_out10 = Activation('sigmoid')(spec_resblock10)

    spec_act10= keras.layers.Multiply()([tanh_out10, sigm_out10])
       
    spec_drop10 = keras.layers.SpatialDropout1D(rate=0.5)(spec_act10)
    
    skip_output10= QuaternionConv1D(filters=32, kernel_size=(1),padding='same')(spec_drop10)

    res_output10 = keras.layers.Add()([res_output9, skip_output10])
    
    
    
    
    #####


     # Residual blocks sum


    resblock_sum=keras.layers.Add()([skip_output1,skip_output2,skip_output3,skip_output4,skip_output5,skip_output6,skip_output7,skip_output8,skip_output9,skip_output10])      
    
    resblock_sum = Activation('relu')(resblock_sum)

    # 1D convolution of 32 filters
    tcn_conv1d_2 = QuaternionConv1D(filters=32,kernel_size=(1), padding='same')(resblock_sum)
    tcn_conv1d_2 = Activation('relu')(tcn_conv1d_2)

    # 1D convolution of 32 filters
    spec_tcn = QuaternionConv1D(filters=32,kernel_size=(1),padding='same')(tcn_conv1d_2)
    spec_tcn = Activation('tanh')(spec_tcn)






    # SED
    #sed = spec_rnn
    sed=spec_tcn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(QuaternionDense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)
    
    # DOA
    #doa = spec_rnn
    doa=spec_tcn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(QuaternionDense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)



    

    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)

    if (summary == True):
        model.summary()

    return model


    
        

    
    
    
    