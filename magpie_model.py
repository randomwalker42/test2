#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:20:03 2018

@author: wangergou
"""

import pandas as pd
from magpie import Magpie


def data_prep(address):
    
    df = pd.read_csv('/home/ubuntu/toxic/train.csv')
    
    for count in range(len(df)):
        
        Id = df.iloc[count].id
        Text = df.iloc[count].comment_text
        
        label = ""
        
        if df.iloc[count].toxic == 1:
            label = 'toxic'
            
        if df.iloc[count].severe_toxic == 1:
            label = 'severe_toxic'
            
        if df.iloc[count].obscene == 1:
            label = 'obscene'
            
        if df.iloc[count].threat == 1:
            label = 'threat'
            
        if df.iloc[count].insult == 1:
            label = 'insult'
            
        if df.iloc[count].identity_hate == 1:
            label = 'identity_hate'
        
        with open(address + '/' + Id + '.txt', "a") as file:
            file.write(Text)
            
        with open(address + '/' + Id + '.lab', "a") as file:
            file.write(label)
            
        print("Data generation finished.")
        

address = "/home/ubuntu/toxic/magpie_data"
    
#data_prep("/Users/wangergou/Downloads/kaggle/Toxic_Comment_Classification/Magpie/data/")

data_prep(address)
    
magpie = Magpie()

print("Loading word vector... \n")

magpie.train_word2vec(address, vec_dim=100)

print("Initializing data... \n")

magpie.init_word_vectors(address, vec_dim=100)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("Training starts... \n")

magpie.train(address, labels, test_ratio=0.2, epochs=30)

magpie.save_model('/home/ubuntu/toxic/magpie_model.h5')


