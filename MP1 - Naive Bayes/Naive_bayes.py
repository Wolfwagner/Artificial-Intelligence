# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)

    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=1.0, pos_prior=0.8, silently=False):
    print_values(laplace,pos_prior)
    positive_counter={}
    negative_counter={}
    pos_rev_list=[]
    neg_rev_list=[]

    count=0
    
    for indiv_list in train_set:
        count=count+1
       
        if train_labels[count-1]==1:
            pos_rev_list = pos_rev_list+ indiv_list
        else:
            neg_rev_list= neg_rev_list+ indiv_list

    for word in pos_rev_list:
        if word not in positive_counter.keys():
            positive_counter[word] = 1
        else:
            positive_counter[word]= positive_counter[word]+1
    

    for word in neg_rev_list:
        if word not in negative_counter.keys():
            negative_counter[word] = 1
        else:
            negative_counter[word]= negative_counter[word]+1
    
    prob_list_pos={}
    prob_list_neg={}

    n_pos=len(pos_rev_list)
    n_neg=len(neg_rev_list)
    V_pos=len(positive_counter)
    V_neg=len(negative_counter)

    for word in pos_rev_list:
        prob_list_pos[word]=(positive_counter[word]+laplace)/(n_pos+laplace*(V_pos+1))
    for word in neg_rev_list:
        prob_list_neg[word]=(negative_counter[word]+laplace)/(n_neg+laplace*(V_neg+1))
    

    
    prob_list_pos["Unknown"]=laplace/(n_pos+laplace*(V_pos+1))
    prob_list_neg["Unknown"]=laplace/(n_neg+laplace*(V_neg+1))
    
    yhats = []
    
    for doc in tqdm(dev_set, disable=silently):
        pos_prob=0
        neg_prob=0
        for dev_word in doc:
            if dev_word not in prob_list_pos.keys():
                pos_prob= pos_prob+math.log(prob_list_pos["Unknown"])
            else:
                pos_prob=pos_prob+math.log(prob_list_pos[dev_word])

            if dev_word not in prob_list_neg.keys():
                neg_prob = neg_prob+math.log(prob_list_neg["Unknown"])
            else:
                neg_prob = neg_prob+math.log(prob_list_neg[dev_word])
        pos_prob=pos_prob+math.log(pos_prior)
        neg_prob=neg_prob+math.log(1-pos_prior)
     
        if pos_prob>= neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats