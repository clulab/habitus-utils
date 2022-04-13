#!/usr/bin/env python
# coding: utf-8

# # Introduction to CGAP data objects #
#
# CGAP data universe begins with the original 18 CGAP surveys (three surveys for each of six countries).  The data are cleaned and aligned, and secondary data products are built, as shown in the following data map.  Most of these data products are flat files that lack metadata.  This notebook describes my first attempt at a json-based representation of data objects.

# In[1]:


import sys
import os, json
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Change this filepath to one for your machine

sys.path.append(
    "/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks"
)


from CGAP_JSON_Encoders_Decoders import (
    Question_Decoder,
    CGAP_Encoded,
    CGAP_Decoded,
    Country_Decoded,
)


# Change this filepath to one for your machine. The actual file is on our Box
# folder at https://pitt.app.box.com/folder/136317983622

Data = CGAP_Decoded()
# Data.read_and_decode('/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/CGAP/Data/Data Objects/CGAP_JSON.txt')
Data.read_and_decode(
    "/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt"
)

countries = ["bgd", "cdi", "moz", "nga", "tan", "uga"]


# Before you begin, go to https://pitt.app.box.com/folder/134211910534 and open one of the User Guides.  Scroll down to
#  page 30 or thereabouts -- it varies by country -- and familiarize yourself with the three surveys (Household,
# Multiple-respondent and Single-respondent).  Note that each survey item has an label such as H28 or A13. These are
# unique: No label appears in more than one survey.  In general, though not without exception, the same question in
# different countries gets the same label.
#
# The `Data` object holds decoded json strings that hold data and metadata for each question.  `Data` is actually a
#  Python SimpleNamespace, which makes it easy to access data using dot notation.   Thus, `Data.__dict__.keys()`
# gives you all the question keys:

# In[2]:


Data.__dict__.keys()


# The syntax of data object identifiers is simply `country+'_'+label`. If you ask for, say, `Data.uga_A1`,
# you get all the metadata for question `A1` and also the data -- the answers in the Uganda survey to question `A1`:

# In[3]:


Data.uga_A1


# Question A1 asks "What is the form of ownership of your land?".  It is a single-answer question (`qtype = 'single'`)
#  which means it expects a single answer that is coded as an integer between 1 and 5. The mapping from answers to
# integers is described in the `answers` dict.
#
# Look at question A1 in any User Guide and you'll see that the text strings in the `answers` dict are not identical
# with the actual answers in the User Guide, but are short mnemonics.
#
# The other metadata fields are
#
# - `label`: the unique identifier of a question
# - `country` : an identifier of the country from which the data is sourced
# - `survey` : refers to intermediate data products (see data map, above).
# - `df` : a pandas dataframe that holds the data, itself.
#
# We'll look at the `df` dataframes after I introduce single- and multiple-answer questions.
#

# ### CGAP_Decoded API ###
#
# CGAP_Decoded objects have several methods for viewing data and meta-data and assembling dataframes from data objects.
#   For a CGAP_Decoded object called Data, these methods are:
#
# - `decode(jstring)` : decodes a JSON string representation of data object(s)

# - `read_and_decode(file)` : reads JSON from file and decodes it

# - `by_name(name)` : convenience function to get a data object by its name; equivalent to Data.__dict__.get(name)

# - `add_col(country, label, df, qtype = 'single', text = None, answers = None)` : adds a column to Data. This is used
# to add derived variables (e.g., a log transform of a variable, or cluster labels, etc.) See section on `add_col`, below.

# - `describe(label, country = 'bgd', display = True)` : prints a short text description of a variable given its label.
#  If display=False, this returns an f-string representation of the description.

# - `col(country,label,*column)`: the workhorse method for assembling pandas dataframes from data objects.
# Given a country and a label, this returns a dataframe. The optional `*column` argument is for multi-answer questions;
#  see below.


# - `col_from_countries(var, countries)` : This assembles a dataframe for one variable from all the specified countries.
#  var is either a label, alone, if the label denotes a single-answer question, or a (label, column_name) tuple if the
# label denotes a multi-answer question. For example, `col_from_countries(('A5','Rice'),['bgd','nga'])` assembles a
# dataframe of the `Rice` answer to question `A5` for Bangladesh and Nigeria.

# - `cols_from_countries(*vars,countries)`: Assembles a dataframe for all the variables for all the countries.
#
#
# Most of these methods are described in some detail later in this Introduction.

# ### Countries don't always have the same answers to questions ###
#
# One tricky aspect of the CGAP data is that a given question won't always have the same answers in
# different countries.  Here's a description of `A1` for Uganda:

# In[4]:


Data.describe("A1", "uga")


# Bangladesh has an additional answer -- Kott -- which it codes as 5:

# In[5]:


Data.describe("A1", "bgd")


# So 5 means 'other' in Uganda's survey and 'Kott' in Bangladesh's.   Sometimes this variability in coding schemes is
# so extreme that it's worth recoding the individual country data to a inter-country scheme (crops and livestock are a
#  good example, see below), but in most cases, like question `A1`, I have not tried to recode all answers to an
# inter-country scheme, so you should be attentive to the `answers` dict and the mapping of answers to numeric codes.
#
# Sometimes, answers aren't recoded but simply recorded "as is", in which case there won't be an `answers` dict.
# For example, question `A2` asks how much land is owned:

# In[6]:


Data.describe("A2", "bgd")


# ## Single- and multiple-answer questions ##
#
# Attributes such as farm size or family size have just one answer per household, but attributes such as the crops
# grown by a household can have several values. There's no easy way to code multiple answers to one question in a
# single variable, so CGAP uses multiple variables to code the answers to multi-answer questions. We've seen examples
#  of single-answer questions above, now let's look at multi-answer questions, such as `H17`, which asks how
# important it is to save money for various purposes:

# In[7]:


Data.moz_H17


# The Data object for `H17` for Mozambique has a field called `column_dict` that you won't see in single-answer Data
#  objects, and the `df` field contains a multi-column pandas dataframe. Each of these columns contains numbers that
# encode the answers described in the `answers` dict; that is, 1, 2 or 3 depending on whether the answer is
# "very_important", "somewhat important" or "not important", respectively.
#
# Look at the first row of the dataframe:  It tells us that the survey respondent from the household with ID 21948170
# thinks it is very important to save for all four kinds of expenses; whereas the second household thinls it is only
# somewhat important to save for future purchases and regular purchases.

# The dataframe for a multi-answer question has one column per answer, and the names of the columns are the mnemonic
#  strings in `column_dict`.  Thus, you can easily find out which farmers in Uganda grow tomatoes:

# In[8]:


Data.moz_A5.df.Tomatoes


# Let me break down this query:
#
# - `Data` is a namespace
# - `moz_A5` is a key into `Data.__dict__` that returns an object that represents question `A5` for Mozambique
# - `df` is the field of the object that contains a pandas dataframe
# - `Tomatoes` is a column in this dataframe

# The df columns for multi-answer questions are always mnemonics for answers.  This is for two reasons:
#  It's easier for the user to ask for `df.Tomatoes` than, say, `df.A5_27`. More importantly, while `df.A5_27`
#  contains data about tomato-growing in Mozambique, it contains data about growing sesame in Bangladesh and
# sugarcane in Tanzania.  These differences in coding have all been resolved behind the scenes.
#
# You can get the same data using the `col` method, described below.

# In[9]:


Data.col("moz", "A5", "Tomatoes")


# ## Household IDs and DataFrame Joins ##
#
# You'll notice that the dataframes associated with Data objects have household ids as their indices.
#  For example, the first record in any Mozambique Data object can be obtained in the usual pandas way:

# In[10]:


Data.moz_A5.df.loc[22552580]


# This is answers to question `A5` for the household with `HHID` 22552580.  Similarly, here is the answer
#  to the earlier question about farm size:

# In[11]:


Data.moz_A2.df.loc[22552580]


# Because Data object `df` indices are unique household IDs, it is straightforward to join `df`s:

# In[12]:


x = Data.moz_A2.df.join(Data.moz_A5.df)
x


# Here, the first column, `A2` is the answers to question `A2` and the rest of the dataframe is the answers
# to questions about crops.

# ## Country-specific Decoded objects ##
#
# If you want to work with data from a specific country, you can make the notation even simpler by using
# the `Country_Decoded` subclass of `CGAP_Decoded` that's specific to a country:

# In[13]:


bgd = Country_Decoded("bgd", Data)
cdi = Country_Decoded("cdi", Data)
moz = Country_Decoded("moz", Data)
nga = Country_Decoded("nga", Data)
tan = Country_Decoded("tan", Data)
uga = Country_Decoded("uga", Data)


# Now you don't have to specify the country as a string:

# In[14]:


print(Data.moz_A5.df.Rice.value_counts())

print(moz.A5.df.Rice.value_counts())

print(moz.col("A5", "Rice").value_counts())

pd.concat([moz.col("A5", "Rice"), moz.col("H28")], axis=1)


# ## Manipulating data ##
#
# This section introduces three methods of `CGAP_Decoded objects`.  The workhorse is `col`, which gets a column of data given a specification, if possible.  `col` takes a country as an argument, but sometimes you will want to get a column of data for one variable for some or all countries; for this, use `col_from_countries`.  To get columns of data for several variables over some or all countries, use `cols_from_countries`.  And be sure to read the warning at the end of this section!

# ### col ###
#
# `col` is a `CGAP_Decoded` method that returns a pandas Series. It is there only for convenience, as it's easy to forget how to access columns, particularly those for single-answer questions.
#
# `Data.col('moz','A5','Rice')` is equivalent to `Data.moz_A5.df.Rice`
#
# `Data.col('moz','H28')` is equivalent to `Data.moz_H28.df.H28`

# In[15]:


# single-answer question:
print(Data.col("moz", "H28"))
print()


# In[16]:


# multi-answer question:
print(Data.col("moz", "A5", "Rice"))
print()


# `col` will provide diagnostic messages and will return `None` when it can't find what you ask for:

# In[17]:


# multi-answer question but the user doesn't know it
print(Data.col("moz", "A5"))


# In[18]:


print(Data.col("moz", "A6", "Wheat"))


# In[19]:


# moz doesn't list wheat as an answer to question A5
Data.col("moz", "A5", "Wheat") is None


# In[20]:


Data.col("moz", "FooBarHaHaHa!") is None


# ### col_from_countries ###
#
# If you need to get the value of a variable for all countries, use `col_from_countries`.  Here are the values of variable `A6` for Bangladesh and Cote d'Ivoire:

# In[21]:


Data.col_from_countries("A6", countries=["bgd", "cdi"])


# That's pretty straightforward, but `A6` is a single-answer question.  What if you want the values over countries for a multi-answer question such as `A5`, which asks which crops are grown? Use a tuple of the question label and the name of the column that holds a particular answer:

# In[22]:


Data.col_from_countries(("A5", "Rice"), countries=["bgd", "cdi"])


# ### cols_from_countries ###
#
# That's fine for single variables, but what if you want a dataframe of one or more variables over several countries? The method `cols_from_countries` builds a dataframe from several variables that can represent single-answer or multiple-answer questions. For example, here are three variables -- number of hectares (A2), whether rice is grown (A5,Rice) and whether maize is grown (A5,Maize) and country for Tanzania and Uganda:

# In[23]:


Data.cols_from_countries(
    "A2", ("A5", "Rice"), ("A5", "Maize"), "COUNTRY", countries=["tan", "uga"]
)


# ### A warning about col_from_countries and cols_from_countries ###
#
# You might recall that `col` tries to give you what you want, but when it can't, it prints a warning and returns `None`.  `col_from_countries` and `cols_from_countries` don't fail when they get `None` from `col`. Instead, they just don't include the countries you think you're getting:

# In[24]:


wheat = Data.col_from_countries(
    ("A5", "Wheat"), countries=["bgd", "cdi", "moz", "nga", "tan", "moz"]
)

print(f"The dataframe includes {len(wheat)} records")


# Instead of records from all the countries, you're getting only the 5994 records from the countries that grow wheat. You can think of this as a bug or a feature! A more troubling case arises when you ask for more than one variable:

# In[25]:


wheat_and_rice = Data.cols_from_countries(
    ("A5", "Wheat"),
    ("A5", "Rice"),
    countries=["bgd", "cdi", "moz", "nga", "tan", "moz"],
)

print(f"The dataframe includes {len(wheat_and_rice)} records")
print(f"The Rice column includes {np.sum(np.isnan(wheat_and_rice['Rice']))} NaNs")
print(f"The Wheat column includes {np.sum(np.isnan(wheat_and_rice['Wheat']))} NaNs")


# What's happened here is that wheat isn't grown in four countries, so the records for households in those countries have NaNs for wheat.  It's the right thing to do, but be aware that it's happening!

# ## Other data manipulation ##
#
# ### Adding new data objects, temporarily ###
#
# You can add new data objects to a CGAP_Decoded object with the `add_col` method.  Here's an example of adding a new variable derived from monthly income (`D21_LZ`) and monthly outgoings (`D19_LZ`):

# In[26]:


for country in countries:
    diff = Data.col(country, "D21_LZ") - Data.col(country, "D19_LZ")
    Data.add_col(
        country=country,
        label="INCOME_DIFF",
        df=diff,
        text="Monthly income minus monthly outgoing",
        qtype="single",
    )


# `text` and `answers` default to `None` and `qtype` defaults to `'single'`, so you can get away with specifying just the positional arguments `country`, `label` and `df`, as in:

# In[27]:


# make a column of random numbers that's the same length as moz
randoms = np.random.random(len(Data.col("moz", "H28")))

# positional arguments are country, label and df
Data.add_col("moz", "random_numbers", randoms)

print(Data.__dict__.get("moz_random_numbers"))


# Note that `add_col` is permissive about what kind of object is passed as `df`:  Anything that can be turned into a pandas dataframe, such as the numpy array `randoms` in the previous example, will work.
#
# #### NOTE ####
#
# `add_col` adds a column temporarily to a CGAP_Decoded object such as `Data`.  At present there's no way to write out the addition permanently.  That's because I wrote the encoder in a way that's too specific to CGAP data.  It's a priority to fix this so that people can save derived variables for CGAP and Manobi data objects.

# ### Other odds and ends (unfinished) ###
#
# Series can be concatenated, provided they have the same indexes:

# In[28]:


df = pd.concat(
    [
        Data.col("moz", "A5", "Rice"),
        Data.col("moz", "H28"),  # "col" version
        Data.moz_H28.df.H28,  # non="col" version: you have to say H28 twice
    ],
    axis=1,
)
df


# Series can also be concatenated with dataframes:

# In[29]:


df = pd.concat([Data.col("moz", "A5", "Rice"), Data.moz_A61.df], axis=1)
df


# ## Some illustrative data analysis (unfinished) ##
#
#
#
# First let's look at the joint distribution of rice and other crops in Cote d'Ivoire:

# In[30]:


x = pd.crosstab(
    Data.col("cdi", "A5", "Rice"), Data.col("cdi", "A5", "Groundnuts"), margins=True
)
x


# We can do the same for multiple countries:

# In[31]:


df = Data.cols_from_countries(
    ("A5", "Rice"), ("A5", "Maize"), countries=["cdi", "moz", "nga"]
)

pd.crosstab(df.Rice, df.Maize)


# This tells us -- among other things -- that growing rice and maize are not independent. For example, if you grow rice in cdi, moz or nga, then the conditional probability of _not_ growing maize is 612/(612+2088) = .23, whereas if you _don't_ grow rice then you probably _will_ grow maize: the conditional probability of growing maize given that you don't grow rice is 3625/(3625+1655)= .69.

# Now for something a bit more challenging:  How do respondents generate income?  Question H6 is a single-answer question that asks about a respondent's primary job, while H2B asks about sources of income:

# In[32]:


print(Data.bgd_H6.text)
print(Data.bgd_H6.answers)
print()
print(Data.bgd_H2B.text)
print(Data.bgd_H2B.answers)


# In[33]:


pd.crosstab(Data.col("bgd", "H6"), Data.col("bgd", "H2B"), normalize="index")


# So, in Bangladesh, among farmers (H6 == 1) the main sources of income are growing crops (71.6% of farmers) or raising livestock (11.4% of farmers).  Laborers (H6 == 5) have the most variable sources of income, with 20% saying they get income from a regular job (H2B ==1), 24% citing occasional work, 16% citing agriculture and 25% citing 'other' as a source of income.
#
# Fewer farmers in Mozambique (H6 == 1) get their income primarily from agriculture (only 45% do). Sixteen percent say they get their income primarily from occasional work (H2B == 2).

# In[34]:


pd.crosstab(Data.col("moz", "H6"), Data.col("moz", "H2B"), normalize="index")


# The raw frequencies are instructive: Among 1870 farming households (76% of those surveyed in moz), 303 say their primary source of income is occasional work and 144 say it is family or friends (H2B == 7).  In fact, 49.7% of the farming households say their primary source of income is not either agriculture (H2B == 7) or raising livestock (H2B==8).  (Exercise for the reader: Contrast Mozambique with Nigeria.)

# In[35]:


pd.crosstab(Data.col("moz", "H6"), Data.col("moz", "H2B"))


# ### Building a classifier with sklearn and data objects ###
#
# The `cols_from_countries` method of CGAP_Decoded objects makes it easy to assemble dataframes for multiple variables for multiple countries. To illustrate, the following class defines a classifier for one or more countries:
#

# In[36]:


import operator
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


class Country_Classifier:
    def __init__(self, cgap_decoded_obj, countries, X, y, model_class, report=False):
        self.Data = cgap_decoded_obj
        self.X, self.y = self.Xy_data(countries, X, y, report)
        self.model_class = model_class

    def Xy_data(self, countries, X, y, report):
        # make one table of X and y so indices remain aligned
        # when we drop rows that contain NaNs
        Xy = self.Data.cols_from_countries(*X, y, countries=countries)
        n0 = len(Xy)

        # drop rows that contain NaNs
        Xy.dropna(axis=0, inplace=True)
        n1 = len(Xy)

        if report:
            print(f"{countries}:  Removed {n0-n1} rows, loss = {(n0-n1)/n0:.4f}\n")

        # reset the index of Xy
        Xy = Xy.reset_index(drop=True)

        # return X and y as separate dataframes. If y denotes a column for a
        # multi-answer then it will be a tuple and the column to drop is the
        # second value in the tuple

        if type(y) in [tuple, list]:
            return Xy.drop(y[1], axis=1), Xy[y[1]]
        else:
            return Xy.drop(y, axis=1), Xy[y]

    def train_model(self):
        self.model = self.model_class.fit(self.X, self.y)

    def score(self, X=None, y=None):
        """ Returns the model score when tested with X and y.  If these are None
        then the default is X = self.X, y = self.y. However, X and y can be new
        data (e.g., from out of sample) """
        X0 = self.X if X is None else X
        y0 = self.y if y is None else y
        return self.model.score(X0, y0)

    def cv_score(self, X=None, y=None, k=5):
        """ Returns the k-fold cross_validated model score when tested with 
        X and y; also returns the scores themselves.  If X and y are None then 
        the default is X = self.X, y = self.y. """
        X0 = self.X if X is None else X
        y0 = self.y if y is None else y
        cv_scores = cross_val_score(self.model, X0, y0, cv=k)
        return np.mean(cv_scores), cv_scores

    def majority_class(self, y=None):
        """ Returns the majority class probability for y and the majority class 
        label.  If y is None then the default is y = self.y. """
        y0 = self.y if y is None else y
        vc = dict(y0.value_counts())
        max_key, max_count = max(vc.items(), key=operator.itemgetter(1))
        return max_count / sum(vc.values()), max_key


ICP = Country_Classifier(
    cgap_decoded_obj=Data,
    countries=["cdi", "nga", "tan"],
    X=["A38", ("A41", "my_legacy"), "NUM_KIDS"],
    y=("A41", "want_children_continue"),
    model_class=RandomForestClassifier(max_features=None),
    report=True,
)


ICP.train_model()
print(f"Model score on training data: {ICP.score()}")
print(f"Cross-validated model score: {ICP.cv_score()}")
print(f"Majority class in training data: {ICP.majority_class()}")


# Here's an analysis that compares predictions in one country given X from that country and a model from another country.

# In[37]:


def intercountry_predictions(cgap_decoded_obj, X, y, model_class, countries):

    results = pd.DataFrame(
        columns=["mc", "self", "self-mc", "n"] + countries, index=countries
    )
    diffs = pd.DataFrame(columns=countries, index=countries)
    country_classifiers = {}

    # First train models for each country on data for that country

    for country in countries:
        CC = Country_Classifier(
            cgap_decoded_obj=cgap_decoded_obj,
            countries=[country],
            X=X,
            y=y,
            model_class=model_class(),
        )
        CC.train_model()
        country_classifiers[country] = CC

        # within-sample classifier score
        results.loc[country, "self"] = round(CC.score(), 3)

        # majority class prediction score
        results.loc[country, "mc"] = round(CC.majority_class()[0], 3)

        # improvement of classifier over majority class
        results.loc[country, "self-mc"] = round(CC.score() - CC.majority_class()[0], 3)

        # number of non-NaN records for available to this classifier
        results.loc[country, "n"] = len(CC.y)

    # Now use these models to predict y's between countries
    for country1 in countries:
        CC1 = country_classifiers[country1]

        for country2 in countries:
            if country1 == country2:
                results.loc[country1, country2] = round(CC1.score(), 3)
                diffs.loc[country1, country2] = 0

            else:
                # Use country1's classifier to predict other countries
                CC2 = country_classifiers[country2]

                # score of country1's  model on c2 data: how well the model predicts c2
                c12 = CC1.score(CC2.X, CC2.y)

                # how much worse it is to use c1 model to predict c2 than it is to use c2 model
                c12_loss = c12 - CC2.score()

                results.loc[country1, country2] = round(c12, 3)
                diffs.loc[country1, country2] = round(c12_loss, 3)

    return results, diffs


X = ["A38", ("A41", "my_legacy"), "NUM_KIDS"]
y = ("A41", "want_children_continue")

results, diffs = intercountry_predictions(Data, X, y, RandomForestClassifier, countries)

print("Accuracy when row country predicts column country\n")
print(results)
print()
print(
    "Loss of accuracy in column country predictions when row country predicts column country\n"
)
print(diffs)


# In[ ]:
