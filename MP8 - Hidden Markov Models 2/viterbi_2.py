"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

epsilon_for_pt = 1e-5
emit_epsilon = 1e-5  

def safe_log(probability, epsilon= 0.00000000000001):
    return math.log(max(probability, epsilon))

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
     
    
    init_prob = defaultdict(lambda: epsilon_for_pt)
    emit_prob = defaultdict(lambda: defaultdict(lambda:epsilon_for_pt ))
    trans_prob = defaultdict(lambda: defaultdict(lambda: epsilon_for_pt))
    
    tag_counts = defaultdict(lambda: 0)
    tag_word_counts = defaultdict(lambda: defaultdict(lambda: 0))
    tag_pair_counts = defaultdict(lambda: defaultdict(lambda:0))
    
    
    word_count={}
    init_tag=[]
    vocab = set() 
    
    total_tags_pairs=0
    
    for sentence in sentences:
        total_tags_pairs=total_tags_pairs+len(sentence)-1
        prev_tag = None  
        init_tag.append(sentence[1][1])
        
        for word, tag in sentence:
            if word not in vocab:
                word_count[word]=1
            else: 
                word_count[word]=word_count[word]+1
                
            tag_counts[tag] = tag_counts[tag]+1
            tag_pair_counts[prev_tag][tag] = tag_pair_counts[prev_tag][tag]  + 1
            tag_word_counts[tag][word] = tag_word_counts[tag][word] + 1
            
            vocab.add(word)
            prev_tag = tag

    no_of_tags = len(tag_counts)-1
    alpha= 1e-7
  
    hapax_counts = Counter()
    hapax_total_count = 0
    
    for word in word_count:
        if word_count[word] == 1:
            hapax_counts.update([word])
            hapax_total_count += 1
                
    hapax_prob = {}
    
    for tag in tag_counts:
        hapax_word_count_for_tag = 0
        for word, tag_count in tag_word_counts[tag].items():
            if word_count[word] == 1:
                hapax_word_count_for_tag += tag_count
            hapax_prob[tag] = hapax_word_count_for_tag / hapax_total_count
               
        
    for tag in tag_counts:

        init_prob[tag] = (init_tag.count(tag)+alpha)/(len(sentences)+alpha*(no_of_tags+1))
        
        unique_words=len(list(tag_word_counts[tag].keys()))
        total_words=0
        
        for word in list(tag_word_counts[tag].keys()):
            total_words= total_words+tag_word_counts[tag][word]
            
        for word in vocab:
            emit_prob[tag][word] = ((tag_word_counts[tag][word] + (alpha* hapax_prob[tag]))/ (total_words+(alpha*hapax_prob[tag])*(unique_words+1)))
            
        
        for next_tag in tag_counts:
            trans_prob[tag][next_tag] = ((tag_pair_counts[tag][next_tag] + alpha) / (tag_counts[tag] + alpha*(no_of_tags+1)))

    for tag in init_prob:
        init_prob[tag] = safe_log(init_prob[tag])

    for tag in emit_prob:
        unique_words=len(list(tag_word_counts[tag].keys()))
        total_words=0
        
        for words in list(tag_word_counts[tag].keys()):
            total_words=total_words+tag_word_counts[tag][words]

        for word in emit_prob[tag]:
            emit_prob[tag][word] = safe_log(emit_prob[tag][word])
        emit_prob[tag]["UNK"] = safe_log((alpha*hapax_prob[tag])/(total_words+(alpha*hapax_prob[tag])*(unique_words+1)))
        
    for tag in trans_prob:
        for next_tag in trans_prob[tag]:
            trans_prob[tag][next_tag] = safe_log(trans_prob[tag][next_tag])
        
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} 
    predict_tag_seq = {} 

    # TODO: (II)
     
    if i == 0:
        for tag in emit_prob:
            if word in emit_prob[tag]:
                log_prob[tag] = prev_prob[tag] + emit_prob[tag][word]
            else:
                log_prob[tag] = prev_prob[tag] + emit_prob[tag]["UNK"] 
            predict_tag_seq[tag] = [tag]
    else:
        for current_tag in emit_prob:
            max_prob = float('-infinity')
            best_prev_tag = None

            for prev_tag in prev_prob:
                transition_prob = trans_prob[prev_tag][current_tag]
                emission_prob = emit_prob[current_tag].get(word, emit_prob[current_tag]["UNK"])

                current_prob = prev_prob[prev_tag] + transition_prob + emission_prob

                if current_prob > max_prob:
                    max_prob = current_prob
                    best_prev_tag = prev_tag

            log_prob[current_tag] = max_prob
            predict_tag_seq[current_tag] = prev_predict_tag_seq[best_prev_tag] + [current_tag]
            
    return log_prob, predict_tag_seq

def viterbi_2(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sentence in test:
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}

        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = init_prob[t]
            else:
                log_prob[t] = safe_log(epsilon_for_pt)
            predict_tag_seq[t] = []
            
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)
            
        
            
        # TODO:(III) 
        
        final_tag = max(log_prob, key=log_prob.get)
        max_tag_sequence = predict_tag_seq[final_tag]

        predicted_sentence = [(word, tag) for word, tag in zip(sentence, max_tag_sequence)]
        predicts.append(predicted_sentence)
     
    return predicts
