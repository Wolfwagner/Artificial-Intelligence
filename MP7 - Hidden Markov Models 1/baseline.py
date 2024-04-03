"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    freq_table = {}
    tag_table = {}
    result = []
    for sentences in train:
        for word, tag in sentences:
            if word == 'START' or word == 'END':
                continue
            if word not in freq_table:
                freq_table[word] = {}
                freq_table[word][tag] = 1
            else:
                if tag not in freq_table[word]:
                    freq_table[word][tag] = 1
                else:
                    freq_table[word][tag] += 1 
            if tag not in tag_table:
                tag_table[tag] = 1
            else:
                tag_table[tag] += 1

    for sentences in test:
        result_sen = []
        for word in sentences:
            if word == 'START' or word == 'END':
                result_sen.append((word, word))
            else:
                if word not in freq_table:
                    tag = max(tag_table, key=tag_table.get)
                    result_sen.append((word, tag))
                else:
                    tag = max(freq_table[word], key=freq_table[word].get)
                    result_sen.append((word, tag))
        result.append(result_sen)

    return result