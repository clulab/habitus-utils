"""
This script takes a folder containing Senegal Actor txt files and extracts sentences that contain
numbers and actor keywords/phrases and converted into TXT.
It works with python3 in a Unix or Linux or (without alarms) Windows system.

Requirements:
    Basic Python3 library
    Spacy library

Usage:
    python3 accumulation_actor_analysis.py input_txt_folder analysis_output_txt search_window_size
"""

import os
import sys
import spacy

#*************************************************************Global Variable Definitions************************************
# load Spacy language model for sentenizer and taggers
nlp = spacy.load("en_core_web_sm")
# since we do not use any parser here, increase the max_length of Spacy
nlp.max_length = 10_000_000 
actors = []
candidate_countries = []
# actors = ["rice", "wife", "wives", "education level", "education", "hectares", "hectare", "loan", "crop", "trade", "peanut"]
# candidate_countries = ["mauritania", "gambia", "cameroon", "guinea bissau", "western sahara", "algeria", "libya", "egypt", "chad", 
#                     "sudan", "central african republic", "morocco", "tunisia", "ghana", "mali", "guinea", "sierra leone", 
#                     "liberia", "niger", "nigeria", "burkina faso", "benin", "congo", "kenya", "ethiopia", "angola", "uganda", 
#                     "somalia", "rwanda", "burundi", "namibia", "south africa", "botswana", "eswatini", "tanzania", "zambia", 
#                     "zimbabwe", "mozambique", "eritrea", "djibouti", "togo", "côte d'ivoire"]
related_countries = {}
target = ""
#******************************************************************************************************************************

'''
    This function takes in the ``text`` to be sentenzied and the size of search windows ``search_window_size`` to search for 
    co-occurence of each pair target-candidate countries.
'''
def extract_sents(text, search_window_size):
    '''
    This function takes in the ``text`` to be sentenzied and the size of search windows ``search_window_size`` to search for 
    co-occurence of each pair target-candidate countries.
    ``text`` here, is a big string of all the sentences.
    '''
    # We will use spacy sentenizer to segment them and check if that sentence is interesting or not
    doc = nlp(text)
    # this is how you can turn doc.sents into list of Spacy sent object 
    # sents = list(doc.sents)
    sent_text = ""
    count = 0
    for sent in doc.sents:
        sent_text += " " + sent.text
        if count % search_window_size == 0:
            countries = isContained_countries_and_actors(sent_text)
            if len(countries) > 0:
                for country in countries:
                    related_countries[country].append(sent_text)
            sent_text = ""
        count += 1
    if sent_text != "":
        countries = isContained_countries_and_actors(sent_text)
        if len(countries) > 0:
            for country in countries:
                related_countries[country].append(sent_text)

# This sent here is string type (after calling sent.text and concatenate them together) for now.
def isContained_countries_and_actors(sent):
    return [country for country in candidate_countries if (country in sent and target in sent)]

'''
   Function name: get_country_similarity_scores
   Parameters:
    1. inputdir - directory where plain text files converted from downloaded pdfs
    2. search_window_size - number of sentences we will search for occurence of target and at least one of the candidate countries.
    3. target_country - the country to which we trying to find similar countries
    4. similar_countries - list of candidate countries
    5. factors
'''    
def get_country_similarity_scores(inputdir, search_window_size, target_country, similar_countries, factors):
    # access global variables inside a function
    global target, actors, candidate_countries, related_countries
    # set target country
    target = target_country
    # set actors
    actors = factors
    # set list of calculating countries
    candidate_countries = similar_countries
    # populate this related countries dictionary
    for country in candidate_countries:
        related_countries[country] = []
    print("Calculating Country Similarity Scores...")
    # go through each of them
    for txtfile in os.listdir(inputdir):
        filename = os.path.splitext(txtfile)[0]
        filedirin = os.path.join(inputdir, txtfile)
        with open(filedirin, 'r') as filein:
            # first we need to remove empty line and add all the text together
            # for now we do not use sentence distance but to use the line that each word appears.
            text = ""
            for line in filein:
                line = line.strip("\n").replace("\t", " ").lower()
                if line != "":
                    text += " " + line
            extract_sents(text.replace("\t", " "), search_window_size)
    similarity_scores = {}
    for country in related_countries:
        #followed by a country name
        similarity_scores[country] = len(related_countries[country])
    return similarity_scores
        