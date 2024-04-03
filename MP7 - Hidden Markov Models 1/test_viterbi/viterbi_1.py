
"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
from collections import defaultdict, Counter

# Small epsilon values for Laplace smoothing
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = {}
    emit_prob = defaultdict(lambda: defaultdict(lambda: emit_epsilon))
    trans_prob = defaultdict(lambda: defaultdict(lambda: emit_epsilon))
    
    tag_counts = defaultdict(lambda: 0)
    tag_pair_counts = defaultdict(lambda: defaultdict(lambda: 0))
    tag_word_counts = defaultdict(lambda: defaultdict(lambda: 0))
    vocabulary = set() 
    word_count={}
    init_tag=[]
    init_tag_dic={}
    total_tags_pairs=0
    for sentence in sentences:
        total_tags_pairs=total_tags_pairs+len(sentence)-1
        prev_tag = 'START'  # Initial tag
        init_tag.append(sentence[1][1])
        for word, tag in sentence:
            if word not in vocabulary:
                word_count[word]=1
            else: 
                word_count[word]=word_count[word]+1
            tag_counts[tag] += 1
            tag_pair_counts[prev_tag][tag] += 1
            tag_word_counts[tag][word] += 1
            vocabulary.add(word)
            prev_tag = tag

    total_tags = len(tag_counts)
    total_words = len(vocabulary)
    alpha_init=0.001
    alpha_tag=0.001
    alpha=0.0001
    for tag in tag_counts:

        init_prob[tag] = (init_tag.count(tag)+alpha_init)/(len(sentences)+alpha_init*(total_tags+1))

        Vt=len(list(tag_word_counts[tag].keys()))
        nt=0
        
        for words in list(tag_word_counts[tag].keys()):
            nt=nt+tag_word_counts[tag][words]
        for word in vocabulary:
            emit_prob[tag][word] = ((tag_word_counts[tag][word] + alpha) / (nt+alpha*(Vt+1)))
        
        for next_tag in tag_counts:
            trans_prob[tag][next_tag] = ((tag_pair_counts[tag][next_tag] + alpha_tag) / (tag_counts[tag] + alpha_tag*(total_tags+1)))

    for tag in init_prob:
        init_prob[tag] = math.log(init_prob[tag])

    for tag in emit_prob:
        Vt=len(list(tag_word_counts[tag].keys()))
        nt=0
        
        for words in list(tag_word_counts[tag].keys()):
            nt=nt+tag_word_counts[tag][words]

        for word in emit_prob[tag]:
            emit_prob[tag][word] = math.log(emit_prob[tag][word])
        emit_prob[tag]["unknown"] = math.log(alpha/(nt+alpha*(Vt+1)))
    for tag in trans_prob:
        for next_tag in trans_prob[tag]:
            trans_prob[tag][next_tag] = math.log(trans_prob[tag][next_tag])
        
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    log_prob = {}  # Store the log_prob for all the tags at the current column (i)
    predict_tag_seq = {}  # Store the tag sequence to reach each tag at column (i)

    if i == 0:
        for tag in emit_prob:
            if word in emit_prob[tag]:
                log_prob[tag] = prev_prob[tag] + emit_prob[tag][word]
            else:
                log_prob[tag] = prev_prob[tag] + emit_epsilon  # Handle unknown words
            predict_tag_seq[tag] = [tag]
    else:
        for current_tag in emit_prob:
            best_prob = float('-inf')
            best_prev_tag = None

            for prev_tag in prev_prob:
                transition_prob = trans_prob[prev_tag][current_tag]
                emission_prob = emit_prob[current_tag].get(word, emit_epsilon)
                
                current_prob = prev_prob[prev_tag] + transition_prob + emission_prob

                if current_prob > best_prob:
                    best_prob = current_prob
                    best_prev_tag = prev_tag

            log_prob[current_tag] = best_prob
            predict_tag_seq[current_tag] = prev_predict_tag_seq[best_prev_tag] + [current_tag]

    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sentence in test:
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}

        # Initialize log probabilities and predicted tag sequences
        for tag in emit_prob:
            if tag in init_prob:
                log_prob[tag] = init_prob[tag]
            else:
                log_prob[tag] = math.log(epsilon_for_pt)
            predict_tag_seq[tag] = []

        # Forward pass to calculate log probabilities for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

        # Backward pass to find the best path through the trellis
        best_final_tag = max(log_prob, key=log_prob.get)
        best_tag_sequence = predict_tag_seq[best_final_tag]

        # Combine words and predicted tags to create the final (word, tag) pairs
        predicted_sentence = [(word, tag) for word, tag in zip(sentence, best_tag_sequence)]
        predicts.append(predicted_sentence)

    return predicts