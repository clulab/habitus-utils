from numpy import random
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-colorblind')
#datefn from:https://stackoverflow.com/questions/39260616/generate-a-normal-distribution-of-dates-within-a-range
import time
import numpy



INPUTS=["XXXX-07-07 -- XXXX-07-22","XXXX-05-15","1358-XX-XX"]

#dates_with_year=INPUT.replace("XXXX", "2017")
#_DATE_RANGE=tuple(dates_with_year.split("--"))


_DATE_FORMAT = '%Y-%m-%d'
_EMPIRICAL_SCALE_RATIO = 0.15
_DISTRIBUTION_SIZE = 1000
_SCALE_RATIO_FOR_DRAWING=1e8
GENERATE_DUMMY_DATA=True

numpy.random.seed(3)

'''
COMMENT FROM MIHAI
1. Since the dates for sowing repeat every year, I think we should not use year values at all. That is, we should only rely on months and days.



if the input is a range()
{for each end points}
clean_and_check()
}

for any input from tsv
check if its date range or date?

if(its a date range:)
{
for each end points
{
string_replace_xxxx()
check_year_only()
}
find timedelta  between end and beginning date
run a for loop from begin date with number of dates to get each date
add to list
return list
}

if its just a date():
{
string_replace_xxxx()
check_year_only()
add to list of dates
}

for x in list of dates:
get_unixtimestamp()
give new list to distribtuion function

------functions
def get_unixtimestamp(date):

}

for each date in this range:

def get_timestamp
{
split out year, month and day from it, 
replace its year with same year for every value
and get a unix time stamp for it?
add to a list of dates
}


def check_if_dattime_passable()
{check if the given string is understood by datetime.strptime()}

def string_replace_xxxx()

{return string replace XXXX  with the same custom value e.g.,2019,}



def check_year_only()
{check if it has only year,  reject, else convert string to date time and return}


2. Dates that include only year information should be removed.
3. The ranges should be expanded to all the dates included.


So, for example, rule 2 indicates that we should NOT use row 8, "1358", and row 17: 2011-XX-XX -- 2018-XX-XX.
Individual dates should be without year, e.g., row 20 becomes simply "XXXX-05-10".
Ranges are expanded, e.g., "between 7 and 22 July" in row 2 becomes: XXXX-07-07, XXXX-07-08, XXXX-07-09, ... , XXXX-07-22.

'''

def check_if_range(input):
    if len(input.split("--"))

for each_input in INPUTS