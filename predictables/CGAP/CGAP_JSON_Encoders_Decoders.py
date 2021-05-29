#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:38:19 2021

@author: prcohen
"""

from types import SimpleNamespace
import os, json

import numpy as np
import pandas as pd
import math

class Question_Encoder ():
    """
    Irrespective of question type, self.text is the text of the question
    and self.qtype is either 'single' (for a single-answer question) or
    'multi' (for a multi-answer question). self.country is one of 'bgd','cdi',
    'moz', 'nga','tan','uga'; and self.survey is one of 'hh' (household), 
    'mr' (multi-respondent), 'rr' (reduced multi-respondent), or
    'sr' (single-respondent). 
    
    With country and survey, we can identify a pandas df (e.g., df = moz_rr).  
    
    If self.qtype = 'single' then self.label is the label of the question  
    (e.g., 'H6').  It is expected that df contains a column with that name. 
    self.answers maps numeric codes to (typically) textual answers 
    (e.g., {1:'farmer',2:'professional',...}). self.answers is optional, it
    really just provides an answer key.  By convention, self.answers = None means 
    that the values in column have not been recoded (e.g., age, village name). 
    
    If self.qtype = 'multi' then self.question_label is the prefix for
    multiple labels (e.g., 'A6' is the prefix for A6_1, A6_2,...). 
    self.column_dict maps symbols that represent the labels of answers of 
    multi-answer questions to the df columns that contain the answers.  
    For example, if question_text is "Do you grow the following
    crops?", then the column dict might contain {'rice' : 1, 'wheat' : 2}, 
    meaning that the columns A6_1 and A6_2 contain the yes/no answers to 
    whether one grows rice and wheat, respectively.  
    """
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)         
            
        self.df_name = self.country+'_'+self.survey  # the name, not the actual df
        
        
    def make_columns (self):
        qdf = pd.DataFrame()
        if self.qtype == 'single':
            col = self.df[self.df_name].get(self.label)
            if col is not None:
                qdf[self.label]  =  col
            else:
                print (f"{self.df_name}.{self.label} does not exist")
        else:
            for k,v in self.column_dict.items():
                col = self.df[self.df_name].get(self.label+'_'+str(v))
                if col is not None:
                    qdf[k] = col
                else:
                    print (f"{self.df_name}.{self.label+'_'+str(v)} does not exist")
        self.columns = qdf.to_json(orient='split')          
    
    def encode(self):
        del self.__dict__['df']
        return json.dumps(self.__dict__)
    


class Question_Decoder (SimpleNamespace):
    def __init__(self,json_string):
        self.__dict__.update(json.loads(json_string))
        self.df = pd.read_json(self.columns,orient='split', convert_axes = False)
        # the convert_axes = False argument seems to protect against a bizarre
        # behavior where pd.read_json reads some strings as datetimes, though they are not.
        del self.columns
            

class CGAP_Encoded ():
    def __init__(self):
        self.jstrings = []
    
    def add_encoded (self,encoded):
        self.jstrings.append(encoded)
    
    def write (self, file): 
        f = open(file, 'w') if os.path.isfile(file) else open(file, "x")
        print(*self.jstrings, sep = '\r', file = f)
        f.close
        

class CGAP_Decoded (SimpleNamespace):
    def __init__(self):
        pass
        
    def decode (self, jstring):
        # this decodes a string and adds it to self.__dict__ with a key
        # made from the country string and question label
        d = Question_Decoder(jstring)
        self.__dict__.update({d.country+'_'+d.label : d})
        return d
           
    def read_and_decode (self, file):
        with open(file,'r') as f:
            for jstring in f.readlines():
                self.decode(jstring)
    
    def by_name (self,name):
        # convenience function to let us construct the name from parts
        # e.g., [self.by_name(country+'_H6') for country in ['bgd','uga']]
        return self.__dict__.get(name)
    
    def df (self, name):
        # convenience function to return the df associated with given a 
        # country_label such as  moz_A1
        return self.by_name(name).df
    
    def add_col (self, country, label, df, qtype = 'single', text = None, answers = None):
        df = pd.DataFrame(df,columns = [label])
        self.__dict__.update(
            {country+'_'+label : 
              SimpleNamespace(
                  qtype = qtype, text = text, answer = answers,
                  country = country, label = label, df = df) 
              })
            
    def describe (self, label, country = 'bgd', display = True):
        """ Given a label such as 'A5', this returns a tuple of f-strings that
        describe the associated data object. If a country is specified, this 
        describes the data associated with the label for that country.  This can
        be useful when the answers or columns (for single- and multi-answer 
        questions, respectively) are different for different across countries.
        For a non-verbose description it will often be sufficient to default
        to 'bgd' as the country."""
        obj = self.__dict__.get(country+'_'+label)
        if obj is None:
            print (f"{country+'_'+label} does not exist")
            return None
        description = [f"{k} : {obj.__dict__.get(k)}\n" for k in
                       ['label','text','qtype','survey','answers']]
        
        cd = obj.__dict__.get('column_dict') 
        if cd is not None:
            description.append(f"column names : {list(cd.keys())}\n")
        
        if display: 
            print(*description)
        else:
            return "\n\n"+"".join(description)
        
         
    def col (self,country,label,*column):  
        """ Returns a pd series (not a df) for the country and label.  If label denotes
        a multi-answer question, a column is expected; for example, col('moz','A5,'Rice') """                                                                  
        obj = self.__dict__.get(country+'_'+label)
        if obj is None:
            print (f"{country+'_'+label} does not exist")
            return None
        if obj.qtype == 'single':
            if column != ():
                print (f"{label} is a single-answer question; ignoring {column[0]}")
            return obj.df.get(label)
        if obj.qtype == 'multi': 
            if column != ():
                c = obj.df.get(column[0])
                if c is None:
                    print (f"{country+'_'+label+'.'+column[0]} does not exist")
                return c
            else:
                print (f"\n{country+'_'+label} is a multi-answer question; please specify an answer column.")
                print (f"\nOptions are {list(obj.column_dict.keys())}")
                
        
    def col_from_countries (self, var, countries):
        """ var is either a label, alone, if the label denotes a single-answer question, 
        or a (label, column_name) tuple if the label denotes a multi-answer question. 
        Either way, this function returns a df of exactly one column, which is the 
        append of this column for all the countries in the order specified."""
        if type(var) in [tuple,list]:
            return pd.DataFrame(pd.concat([self.col(country, var[0],var[1]) for country in countries],axis=0))
        else:
            return pd.DataFrame(pd.concat([self.col(country, var) for country in countries],axis=0))

    def cols_from_countries (self, *vars, countries):
        df = pd.DataFrame()
        df=df.join([self.col_from_countries(var, countries) for var in vars],how='outer')
        return df


class Country_Decoded (CGAP_Decoded):
    """ This is a country-specific version of CGAP_Decoded. We create one instance per
    country. For now, we do it by first creating an instance of CGAP_Decoded for the
    entire dataset and then extracting a country's records from it. country is one of
    'bgd','cdi','moz','nga','tan' and 'uga'. """
    
    def __init__(self, country, full_dataset):
        self.__dict__.update({k.split('_')[1]:v
                              for k,v in full_dataset.__dict__.items() if country in k})

    #for any given country, return the list of all available single qn answers
    def concat_all_single_answer_qns(self, qns_to_avoid):
        df=None
        for k,v in self.__dict__.items() :
            if v.qtype=='single':
                label=v.label
                if (label not in qns_to_avoid):
                    for index,x in enumerate(v.df[label]):
                        #if an answer is nan replace it with -1
                        if math.isnan(x):
                            v.df.at[(index+1),label]=-1
                    df = pd.concat([df, v.df.dropna(axis=0)], axis=1)
        assert df is not None
        return df


    def concat_all_multiple_answer_qns(self, qns_to_avoid):
            df=None
            for k,v in self.__dict__.items() :
                if v.qtype=='multi':
                    label=v.label
                    if (label not in qns_to_avoid):
                          for sub_qn in (v.df):
                              for index2,y in enumerate(v.df[sub_qn]):
                                #if a value is nan replace it with -1
                                if math.isnan(y):
                                  v.df.at[(index2+1),sub_qn]=-1
                          df = pd.concat([df, v.df.dropna(axis=0)], axis=1)
            assert df is not None
            return df



