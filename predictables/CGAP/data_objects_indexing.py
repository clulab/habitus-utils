#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 22:17:43 2021

@author: prcohen
"""

import string
import numpy as np
index = {}
stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", 
             "against", "all", "almost", "alone", "along", "already", "also","although",
             "always","am","among", "amongst", "amoungst", "amount",  "an", "and", 
             "another", "any","anyhow","anyone","anything","anyway", "anywhere", 
             "are", "around", "as",  "at", "back","be","became", "because","become",
             "becomes", "becoming", "been", "before", "beforehand", "behind", "being", 
             "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom",
             "but", "by", "call", "can", "cannot", "cant", "co", "come", "comes", "con", "could", "couldnt", 
             "cry", "de", "describe", "detail", "did", "do", "does", "done", "down", "due", "during", "each", 
             "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", 
             "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", 
             "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", 
             "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", 
             "give", "go", "had", "happen", "happens", "has", "hasnt", "have", "he", "hence", "her", "here", 
             "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", 
             "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", 
             "indeed", "interest", "into", "is", "it", "its", "itself", "keep", 
             "last", "latter", "latterly", "least", "less", "ltd", "made", "many", 
             "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", 
             "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", 
             "neither", "never", "nevertheless", "next", "nine", "no", "nobody", 
             "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", 
             "off", "often", "on", "once", "one", "only", "onto", "or", "other", 
             "others", "otherwise", "our", "ours", "ourselves", "out", "over", 
             "own","part", "per", "perhaps", "please", "put", "rather", "re", 
             "same", "see", "seem", "seemed", "seeming", "seems", "serious", 
             "several", "she", "should", "show", "side", "since", "sincere", 
             "six", "sixty", "so", "some", "somehow", "someone", "something", 
             "sometime", "sometimes", "somewhere", "still", "such", "take", "ten", 
             "than", "that", "the", "their", "them", "themselves", "then", "thence", 
             "there", "thereafter", "thereby", "therefore", "therein", "thereupon", 
             "these", "they", "thick", "thin", "third", "this", "those", "though", 
             "three", "through", "throughout", "thru", "thus", "to", "together", 
             "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", 
             "until", "up", "upon", "us", "use", "used", "usually", "very", "via", 
             "was", "we", "well", "were", 
             "what", "whatever", "when", "whence", "whenever", "where", "whereafter", 
             "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", 
             "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", 
             "why", "will", "with", "within", "without", "would", "yes", "yet", "you", "your", 
             "yours", "yourself", "yourselves", "the"]

def flatten_strs(foo):
    for x in foo:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in flatten_strs(x):
                yield y
        else:
            yield x
            

def extract_words (cont):
    w = []
    # extract individual words
    if type(cont) == str:
        w.extend([item.split('_') for item in cont.split()])
    elif type (cont) in [list,tuple]:
        w.extend([extract_words(item) for item in cont])
    elif type (cont) == dict:
        w.extend([item.split('_') for item in flatten_strs(cont)
                  if type(item) == str])
    elif np.isnan(cont): 
        pass # we don't want nans as search terms! 
    else:
        print(f"{cont} must be a string, a list/tuple or a dict")
        return
    # flatten and remove duplicates
    w = set(flatten_strs(w))
    # remove punctuation
    w = [s.translate(str.maketrans('', '', string.punctuation)) for s in w]
    return w
    
