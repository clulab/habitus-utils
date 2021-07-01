#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:13:39 2021

@author: prcohen
"""

import sys, os, json
from types import SimpleNamespace
import string, copy

import numpy as np
import pandas as pd

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

sys.path.append('/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/Data Objects/Code and Notebooks')
from data_objects_indexing import extract_words, flatten_strs, stopwords

#==============================================================================


class DO_Encoder ():
    """ A question encoder builds a JSON representation of a data object (DO), 
    which is a collection of data and metadata.  Every question needs a unique
    question id (`name`).  The parameter `data` is required but it doesn't have 
    to be a pandas dataframe, just something that can be turned into one.  
    DOs keep data in a dataframe called `df`. 
    
    If column_dict is None, then D0.df = data; that is, df is simply a copy of the 
    passed data.  If column_dict is not None, then column_dict.keys() says which 
    columns of data are added to DO.df, and column_dict.values() specifies what 
    those columns should be called in DO.df. 
    """
    def __init__(self, name, data, column_dict = None, **kwargs):
        for k,v in kwargs.items(): setattr(self,k,v)
        
        self.name = name
        self.column_dict = column_dict
        
        if type(data) == pd.DataFrame:
            if self.column_dict is None:
                self.df = copy.copy(data)
            else:
                # self.df is a selection of columns in data specified by column_dict
                self.df = pd.DataFrame({v : data.get(k) for k,v in self.column_dict.items() 
                                    if  data.get(k) is not None})
        elif type(data) == pd.Series:
            if self.column_dict is None:
                self.df = pd.DataFrame(data,columns=[data.name])
            else:
                # self.df is the column assigned a name by column_dict
                self.df = pd.DataFrame(data,columns=self.column_dict[data.name])
        else:
            try:
                # try turning data into a DataFrame
                self.df = pd.DataFrame(data)
                if self.column_dict is not None:
                    self.df = pd.DataFrame({v : self.df.get(k) for k,v in self.column_dict.items() 
                                    if  self.df.get(k) is not None})      
            except:
                print ("Can't turn data into a pandas dataframe.")
        
        
    
    def encode(self, make_search_terms = True):
        """ This encodes the DO_Encoder as a json string.  By default
        it creates search terms first, so that the question can be found
        later by keyword search. If this is done before overwriting df, then
        the search terms will include df column names."""
        
        if make_search_terms and self.__dict__.get('search_terms') is None:
            self.make_search_terms()
            
        # encode df to JSON and overwrite the df to free up space
        self.df = self.df.to_json(orient='split')     
        
        return json.dumps(self.__dict__)
    
    def make_search_terms (self, *attributes):
        """ A corpus of questions can be searched by keyword, so this method
        runs the Porter stemmer over all the content words in the question and
        stores both the words and the stems for future indexing. The default 
        is to use all the attributes of the question (e.g., text, answers, notes, 
        etc), but specific attributes can be targeted if *attributes is not None. 
        """
        if not 'nltk' in sys.modules:
            print ("make_stems requires nltk to be installed")
            return
        
        if attributes is not ():
            fields = attributes    
        else:
            # use all the attributes except df
            fields = [x for x in self.__dict__.keys() if x != 'df']
        
        # some attributes aren't specified, so get only the attribute values that aren't None
        parts = [item for item in [self.__dict__.get(field) for field in fields] if item is not None]
        
        # if df hasn't already been encoded to JSON, we can add column names
        # as search terms
        if type(self.__dict__.get('df')) != str:
                parts.extend(self.__dict__.get('df').columns)
        
        # Now extract words and remove punctuation.  Ordinarily one would also lowercase the
        # words, but some  of them may be variable names, so it's best not to. The stemmer
        # in the next step will lowercase, so there's no harm in leaving the words themselves
        # in their original cases.
        acc = [extract_words (part) for part in parts]
        acc = [word for word in set(flatten_strs(acc)) if word is not None]
        
        if acc != []:
            # remove stopwords
            acc = [word for word in acc if not word.lower() in stopwords]

            # Now run the stemmer:
            ps = PorterStemmer()
            stemmed = [ps.stem(word) for word in acc]
            acc.extend(stemmed)

            # remove duplicates again (stems are often the words themselves), 
            # then save the words and stems 
            self.search_terms = list(set(acc))
        else:
            self.search_terms = []
            

#==============================================================================

class DO_Decoder (SimpleNamespace):
    def __init__(self,json_string):
        self.__dict__.update(json.loads(json_string))
        self.df = pd.read_json(self.__dict__.get('df'),orient='split', convert_axes = False)
        # the convert_axes = False argument seems to protect against a bizarre
        # behavior where pd.read_json reads some strings as datetimes, though they are not.
    
    def describe (self, *attributes, display = True):
        """ This returns a tuple of f-strings that describe the decoded data object. 
        If attributes is None, this will describe all attributes other than df
        and the search terms. If display = False, this will return the f-string 
        representation of the description."""
        
        if attributes is ():
            to_describe = [x for x in self.__dict__.keys() if x not in ['df', 'search_terms']]
        else:
            to_describe = attributes
            
        description = [f"{k} : {self.__dict__.get(k)}\n" for k in to_describe]
        
        cd = self.__dict__.get('column_dict') 
        if cd is not None:
            description.append(f"column names : {list(cd.keys())}\n")
        
        if display: 
            print("\n"+"".join(description))
        else:
            return "\n"+"".join(description)
        
    def cols (self, *columns):
        """ Returns the df columns in *cols. """
        
        df = self.__dict__.get('df')
              
        if df is None:
            print (f"Sorry, {self.name} has no dataframe.")
            return
        
        if columns is ():
            return df
        else:
            # check that the specified columns exist in df and return those that do
            # and warn if any don't
            valid_cols = [c for c in columns if c in df.columns]
            if len(valid_cols) < len(columns):
                print (f"At least one of the columns you requested isn't in {self.name} dataframe")
        
        return df[valid_cols]
    
    
    def re_encode (self,*new_name):
        """ This returns a new DO_Encoder object for a decoded object. 
        It's useful when we want to decode a DO to edit it, then recode it.
        The optional argument new_name assigns a new question id to the recoded
        object; it's analogous to `save as`.  Any changes to the decoded object, 
        including changes to its dataframe, are re-encoded.
        
        One consequence is that, while a column_dict might have been used to 
        build the dataframe of original data object, it should not be included 
        when the object is re-encoded. To see why, imagine column_dict = {'x' : 'y'} 
        was used to build the original object.  Once built and decoded its df 
        will have a column called y. But if the column_dict is used in the 
        re_encoding, then the encoder will try to build a new df. It'll look 
        in the original df for a column called x, which doesn't exist. 
        """
        
        return DO_Encoder (
            name = new_name[0] if new_name else self.__dict__.get('name'),
            data = self.__dict__.get('df'),
            column_dict = None, # see comment above
            **{k:v for k,v in self.__dict__.items()
               if k not in ['name','df','column_dict']} )
    
#==============================================================================
   
class Encoded_DOs ():
    def __init__(self):
        self.jstrings = []
    
    def add_encoded (self,encoded):
        if type(encoded) in [list,tuple]:
             self.jstrings.extend(encoded)
        else:
            self.jstrings.append(encoded)
    
    def write (self, file): 
        f = open(file, 'w') if os.path.isfile(file) else open(file, "x")
        print(*self.jstrings, sep = '\r', file = f)
        f.close       
        
#==============================================================================
        
class Decoded_DOs (SimpleNamespace):
    def __init__(self):
        self.term_index = None
        
        
    def decode (self, jstring):
        # this decodes a string and adds it to self.__dict__ .  The key is
        # the question id, `name`. 
        d = DO_Decoder(jstring)
        self.__dict__.update({d.name : d})
        return d
           
    def read_and_decode (self, file):
        with open(file,'r') as f:
            for jstring in f.readlines():
                self.decode(jstring)
    
                
    def build_term_index (self):
        self.term_index = {}
        for obj in self.__dict__.values():
            if type(obj) == DO_Decoder:
                name = obj.name
                if obj.search_terms is not None:
                    for term in obj.search_terms:
                        i = self.term_index.get(term)
                        if i is None:
                            self.term_index[term] = set([name])
                        else:
                            if name not in i:
                                self.term_index[term].update([name])
                            
                        
    def search (self, *query, display = True):
        """ Simple search method to find questions that contain keywords.
        Conjunctive queries are wrapped in parentheses; for example,
        search ('cash',('money','bank')) finds all questions that contain
        'cash' OR ('money' AND 'bank').  This requires self.term_index, so if it
        doesn't exist, the first step is to build it.
        
        The terms in the query are stemmed.
        
        """
            
        if self.term_index is None:
            print ("Building a term index...")
            self.build_term_index()
            
        ps = PorterStemmer()
            
        acc = []
        for term in query:
            if type(term) == tuple:
                # all terms or their stemmed equivalents must be in term_index
                # the stemmed version is more permissive, so it is checked first
                parts = [
                     self.term_index.get(ps.stem(item)) or self.term_index.get(item)
                    for item in term]
               
                # if None in parts then some word in the conjunct isn't found
                if None not in parts:
                    acc.extend (parts[0].intersection(*parts[1:]))
            else: 
                # the term or its stemmed equivalent must be in term_index
                x = self.term_index.get(ps.stem(term)) or self.term_index.get(term)
                if x is not None:
                    acc.extend (x)
        
        acc = set(acc) # remove duplicates
        
        if len(acc) > 0:
            
            if display is True:
                for q in acc: self.describe(q)   
            
            elif type(display) in [list,tuple]:
                for q in acc: self.describe(q, *display)
            
            else:
                return acc
        else:
                print("No items found")
    
    def dob (self,name):
        """ Returns a data object with the given name if it exists, otherwise 
        warns user and returns None. """
        DO = self.__dict__.get(name)
        if DO is not None:
            return DO
        else:
            return print (f"{name} cannot be found")
    
    
    def describe (self, name, *attributes, display = True):
        """ Given a question id (`name`) this returns a tuple of f-strings that
        describe the associated data object. If attributes is None, this
        will describe all attributes other than df."""
        dob = self.dob(name) 
        if dob is None:
            print (f"{name} does not exist")
            return None
        else:
            dob.describe(*attributes,display=display)

    def re_encode (self):
        """ This re-encodes all the decoded questions in self into a newly
        created Encoded_DOs object, which it returns.  The new Encoded_DOs
        object can then be written to a file.  Re-encoding is used after a set of
        decoded questions has been edited or augmented with new questions and
        needs to be saved to a file. 
        """
        New_Encoded = Encoded_DOs()
        for name in self.__dict__.keys():
            if name != 'term_index':
                New_Encoded.add_encoded(self.dob(name).re_encode().encode())
        return New_Encoded
            
    
    def cols (self, *columns):  
        """ 
        This assembles a dataframe from the columns in *columns.  The columns must have 
        the same index because they will be inner-joined.  The syntax for *columns 
        must distinguish single- and multi-answer questions.  Single-answer 
        questions are referred to by the question id, but multi-answer questions 
        are referred to by tuples in which the first element is the question id 
        and the other elements are answer columns.  For example, cols('A6',('A5','Rice')) 
        returns a dataframe comprising the data associated with A6 and the Rice answer
        column to question A5.
        """ 
    
        X = pd.DataFrame()
        
        for item in columns:
            # the first df to be added to assembled sets the index for assembled 
            join_method = 'right' if X.size == 0 else 'inner'
                
             
            if type(item) in [tuple,list]:
                dob = self.dob(item[0])
                if dob is not None: 
                    X = X.join(dob.cols(*item[1:]), how = join_method)
            else:
                dob = self.dob(item)
                if dob is not None:
                    X = X.join(dob.cols(), how = join_method)
                    
        return X
                
#==============================================================================
        
class Decoded_CGAP_DOs (Decoded_DOs):
    def __init__(self):
        super().__init__()
        
        self.names = None
    
    def cols_from_countries (self,*cols,countries):
        
        if self.names is None: 
            # on the first call to cols_from_countries, make a set of the 
            # question ids minus their country codes
            self.names = set(["".join(name.split('_')[1:]) for name in self.__dict__.keys() if name != 'term_index'])
        
        def countrify (names,country):
            """ When cols_from_countries is called, its arguments will be names
            without country prefixes (e.g., 'A6' or ('A5','Rice') ) .  These 
            need to be transformed into names with country prefixes. This requires
            us to distinguish names like 'A5' from answer-column names like 'Rice'.
            This is done by consulting self.names."""
            
            c = lambda country, x : country+'_'+x if x in self.names else x
            
            return [c(country,x) if type(x) not in [list,tuple] else [c(country,y) for y in x] for x in names]
                
        
        acc = []
        for country in countries:
            acc.append( self.cols ( *countrify(cols,country) ) )
            
        return pd.concat(acc)
        