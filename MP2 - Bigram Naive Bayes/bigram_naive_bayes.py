# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023

import reader
import math
from collections import Counter

def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, yhats = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, yhats

def getWordCount(train_set, train_labels, isPos):
    word_count = {}

    for i in range(len(train_labels)):
        if (train_labels[i] != isPos):
            continue
        cur = train_set[i]

        for word in cur:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    return word_count

def getBigramMap(train_set, train_labels, isPos):
    bigram_map = {}
    for i in range(len(train_labels)):
        if train_labels[i] != isPos:
            continue
        cur = train_set[i]
        for j in range(len(cur)-1):
            bg = tuple((cur[j], cur[j+1]))
            if bg in bigram_map:
                bigram_map[bg] += 1
            else:
                bigram_map[bg] = 1
    return bigram_map

def makeProbs (bg_map, laplace):
    probmap = {}
    tot_bg = 0
    tot_types = len(bg_map)

    for bg in bg_map:
        tot_bg += bg_map[bg]

    u_prob = laplace/(tot_bg + laplace*(tot_types))

    for bg in bg_map:
        prob = (bg_map[bg] + laplace)/(tot_bg + laplace*(tot_types))
        probmap[bg] = prob

    return probmap, u_prob

def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.000856, bigram_laplace=0.004643, bigram_lambda=0.7789, pos_prior=0.99, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    pos_words_count_map = getWordCount(train_set, train_labels, 1)
    neg_words_count_map = getWordCount(train_set, train_labels, 0)

    pos_words_probs_map, pos_unknown = makeProbs(pos_words_count_map, unigram_laplace)
    neg_words_probs_map, neg_unknown = makeProbs(neg_words_count_map, unigram_laplace)

    yhats = []
    dev_neg = []
    dev_pos = []

    for review in dev_set:
        pos_p = 0
        neg_p = 0

        for word in review:
            if word in pos_words_probs_map:
                pos_p += math.log(pos_words_probs_map[word])
            else:
                pos_p += math.log(pos_unknown)

            if word in neg_words_probs_map:
                neg_p += math.log(neg_words_probs_map[word])
            else:
                neg_p += math.log(neg_unknown)
        pos_p =pos_p+ math.log(pos_prior)
        neg_p =neg_p+ math.log(1 - pos_prior)

        dev_pos.append(pos_p)
        dev_neg.append(neg_p)

    bigram_pos_count = getBigramMap(train_set, train_labels, 1)
    bigram_neg_count = getBigramMap(train_set, train_labels, 0)

    bigram_pos_probs_map, bi_pos_uk = makeProbs(bigram_pos_count, bigram_laplace)
    bigram_neg_probs_map, bi_neg_uk = makeProbs(bigram_neg_count, bigram_laplace)

    bigram_dev_pos = []
    bigram_dev_neg = []

    for review in dev_set:
        bi_pos_p = 0
        bi_neg_p = 0

        for j in range(len(review) - 1):
            bg = tuple((review[j], review[j + 1]))

            if bg in bigram_pos_probs_map:
                bi_pos_p += math.log(bigram_pos_probs_map[bg])
            else:
                bi_pos_p += math.log(bi_pos_uk)

            if bg in bigram_neg_probs_map:
                bi_neg_p += math.log(bigram_neg_probs_map[bg])
            else:
                bi_neg_p += math.log(bi_neg_uk)
        bi_pos_p += math.log(pos_prior)
        bi_neg_p += math.log(1 - pos_prior)

        bigram_dev_pos.append(bi_pos_p)
        bigram_dev_neg.append(bi_neg_p)

    for i in range(len(dev_set)):
        total_prob_neg = (1 - bigram_lambda) * dev_neg[i] + (bigram_lambda) * bigram_dev_neg[i]
        total_prob_pos = (1 - bigram_lambda) * dev_pos[i] + (bigram_lambda) * bigram_dev_pos[i]

        if total_prob_pos > total_prob_neg:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats