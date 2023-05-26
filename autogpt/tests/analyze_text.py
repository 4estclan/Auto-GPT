import os
import json
import re
import string
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder


FILE_PATH = './krista_paul/tokens_dict.json'
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def write_to_file(file, data):
    file.write(json.dumps(data))



def detect_patterns(token_dict, regex):
    detected_patterns = {}
    for file, tokens in token_dict.items():
        patterns = {}
        for token in tokens:
            match = re.search(regex, token.lower())
            if match:
                if match.group() in patterns:
                    patterns[match.group()].append(token)
                else:
                    patterns[match.group()] = [token]
        if patterns:
            detected_patterns[file] = patterns
    return detected_patterns



def detect_bigrams(token_dict):
    bigram_measures = BigramAssocMeasures()
    detected_bigrams = {}
    for file, tokens in token_dict.items():
        bigram_finder = BigramCollocationFinder.from_words(tokens)
        bigram_finder.apply_freq_filter(2)
        results = bigram_finder.nbest(bigram_measures.raw_freq, 10)
        if results:
            detected_bigrams[file] = results
    return detected_bigrams


if __name__ == '__main__':
    token_dict = load_data(FILE_PATH)
    detected_patterns = detect_patterns(token_dict, r'alleged|assault|attack|hit')
    detected_bigrams = detect_bigrams(token_dict)
    with open('./krista_paul/detected_patterns.json', 'w') as f:
        write_to_file(f, detected_patterns)
    with open('./krista_paul/detected_bigrams.json', 'w') as f:
        write_to_file(f, detected_bigrams)