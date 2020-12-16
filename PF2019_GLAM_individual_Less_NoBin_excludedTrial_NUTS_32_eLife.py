#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 00:20:06 2020

@author: pradyumna
"""

# Perceptual Framing Analysis Script (Perceptual Experiment)
# GLAM model estimation


import glam
import pandas as pd
import numpy as np
import os.path
import seaborn as sns
import arviz as az

import matplotlib.pyplot as plt

import pymc3 as pm

np.random.seed(23) # from random.org

def z_score1(data_all, part_def,z_score_var):
    z_matrix=[]
    z_matrix_aux=[]

    for i in (data_all[part_def].unique()):
        Choicedata = data_all.loc[data_all[part_def] == i]    
    
        pX_A= pd.to_numeric(Choicedata[z_score_var]) 
        pX_zA= (pX_A - np.mean(pX_A))/np.std(pX_A)

        z_matrix_aux= pX_zA.values
    
        for  j in range(len(z_matrix_aux)):    
            z_matrix.append(z_matrix_aux[j])
    return z_matrix

# Individual GLAM estimation and out of sample prediction (Less)

## Load data

# Load data
sufix = '_individual_Less_NoBin_excludedTrial_Gamma-11_NUTS_32_eLife'
data = pd.read_csv('data/PF2019_data/GlamDataPF2019_Less_NoBin_33.csv')

## Reaction times for each participant

participants = data.subject.unique()
f = plt.figure(figsize=(30,30))
order = 1
sns.set_style('white')

for i in data.subject.unique():
    sub={}
    sub['%s' % i] = plt.subplot(int(len(participants)/5+1), 5, order)
    sub['%s' % i].plot()    
    data[(data.subject == i)].rt.hist()
    sub['%s' % i].set_title('participant %s' % i)
    order += 1


data["zrt"] = z_score1(data,'subject',"rt")

## Remove outliers RT

# remove outliers (two criteria)
data1 = data[(data.zrt <= 3) ]
data1 = data1[(data1.rt <= 20000) ]

#participants = data.subject.unique()
#f = plt.figure(figsize=(30,30))
#order = 1
#sns.set_style('white')

#for i in data.subject.unique():
#    sub={}
#    sub['%s' % i] = plt.subplot(int(len(participants)/5+1), 5, order)
#    sub['%s' % i].plot()    
#    data[(data.subject == i)].zrt.hist()
#    sub['%s' % i].set_title('participant %s' % i)
#    order += 1

participants = data1.subject.unique()
f = plt.figure(figsize=(30,30))
order = 1
sns.set_style('white')

for i in data1.subject.unique():
    sub={}
    sub['%s' % i] = plt.subplot(int(len(participants)/5+1), 5, order)
    sub['%s' % i].plot()    
    data1[(data1.subject == i)].rt.hist()
    sub['%s' % i].set_title('participant %s' % i)
    order += 1

# Subset only necessary columns

# Subset only necessary columns
data = data1[['subject', 'trial', 'choice', 'rt',
         'item_value_0', 'item_value_1',
         'gaze_0', 'gaze_1']]
data.head()

## Split data in training and test sets

train_data = pd.DataFrame()
test_data = pd.DataFrame()

for subject in data.subject.unique():
    subject_data = data[data['subject'] == subject].copy().reset_index(drop=True)
    n_trials = len(subject_data)
    
    subject_train = subject_data.iloc[np.arange(0, n_trials, 2)].copy()
    subject_test = subject_data.iloc[np.arange(1, n_trials, 2)].copy()

    test_data = pd.concat([test_data, subject_test])
    train_data = pd.concat([train_data, subject_train])

#test_data.to_csv(str('data/PF2019_data/GlamDataPF2019_preprocessed_test'+sufix+'.csv'))
#train_data.to_csv(str('data/PF2019_data/GlamDataPF2019_preprocessed_train'+sufix+'.csv'))

print('Split data into training ({} trials) and test ({} trials) sets...'.format(len(train_data), len(test_data)))

train_data

## Individual GLAM estimation

### 1. full GLAM

# Fitting full GLAM
print('Fitting full GLAM individually...')

glam_full = glam.GLAM(train_data)

if not os.path.exists(str('results/estimates/glam_PF2019_full'+sufix+'.npy')):
    glam_full.make_model('individual', gamma_bounds=(-1, 1), t0_val=0)
    glam_full.fit(method='NUTS', tune=1000)
else:
    print('  Found old parameter estimates in "results/estimates". Skipping estimation...')
    glam_full.estimates = np.load(str('results/estimates/glam_PF2019_full'+sufix+'.npy'))   

# Save parameter estimates
np.save(str('results/estimates/glam_PF2019_full'+sufix+'.npy'), glam_full.estimates)
pd.DataFrame(glam_full.estimates)

# Estimate Convergence

rhat_gamma =[] 
rhat_v = []
rhat_tau =[] 
rhat_s =[] 

ess_gamma =[] 
ess_v = []
ess_tau =[] 
ess_s =[] 

part_num =  []
for i in range(len(glam_full.trace)):
    model_trace = glam_full.trace[i]
    # estimate rhat param
    rhats_params = az.rhat(model_trace, method="folded")
    rhat_gamma.append(rhats_params.gamma.values)
    rhat_v.append(rhats_params.v.values)
    rhat_tau.append(rhats_params.tau.values)
    rhat_s.append(rhats_params.s.values)
    part_num.append(i)

    # estimate effective sample size
    ess_model = az.ess(model_trace, relative=False)
    ess_gamma.append(ess_model.gamma.values)
    ess_v.append(ess_model.v.values)
    ess_tau.append(ess_model.tau.values)
    ess_s.append(ess_model.s.values)
    
rhats_params_df = pd.DataFrame()
rhats_params_df['gamma'] = rhat_gamma
rhats_params_df['v'] = rhat_v
rhats_params_df['tau'] = rhat_tau
rhats_params_df['s'] = rhat_s
rhats_params_df['part'] = part_num

ess_params_df = pd.DataFrame()
ess_params_df['gamma'] = ess_gamma
ess_params_df['v'] = ess_v
ess_params_df['tau'] = ess_tau
ess_params_df['s'] = ess_s
    

rhats_params_df.to_csv(str('results/convergence/GlamDataFF2018_indiv_rhatsParams'+sufix+'.csv'))
ess_params_df.to_csv(str('results/convergence/GlamDataFF2018_indiv_essParams'+sufix+'.csv'))
