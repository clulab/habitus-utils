from numpy import random
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-colorblind')
#datefn from:https://stackoverflow.com/questions/39260616/generate-a-normal-distribution-of-dates-within-a-range
from datetime import datetime
from datetime import timedelta
import numpy



INPUTS=["XXXX-07-07 -- XXXX-07-22","XXXX-05-15","1358-8-8","1992-08-08 -- 1992-12-12"]

#dates_with_year=INPUT.replace("XXXX", "2017")
#_DATE_RANGE=tuple(dates_with_year.split("--"))


_DATE_FORMAT = '%Y-%m-%d'
_EMPIRICAL_SCALE_RATIO = 0.15
_DISTRIBUTION_SIZE = 1000
_SCALE_RATIO_FOR_DRAWING=1e8
GENERATE_DUMMY_DATA=True
CUSTOM_YEAR=1982
numpy.random.seed(3)

'''
COMMENT FROM MIHAI
1. Since the dates for sowing repeat every year, I think we should not use year values at all. That is, we should only rely on months and days.

2. Dates that include only year information should be removed.
3. The ranges should be expanded to all the dates included.

So, for example, rule 2 indicates that we should NOT use row 8, "1358", and row 17: 2011-XX-XX -- 2018-XX-XX.
Individual dates should be without year, e.g., row 20 becomes simply "XXXX-05-10".
Ranges are expanded, e.g., "between 7 and 22 July" in row 2 becomes: XXXX-07-07, XXXX-07-08, XXXX-07-09, ... , XXXX-07-22.

control flow:


if the input is a range()
{for each end points}
check_year_only()
}

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
{check if it has only year (split by - and check if xxxx),  reject, else convert string to date time and return}


'''

#if the date is something like ""XXXX-07-07 -- XXXX-07-22"" return true
def check_if_range(input):
    splits=input.split("--")
    if len(splits)>1:
        print(f"{input} its a range")
        return True, splits
    return False, None

#if the date is something like "XXXX" or "1993-XX-XX" return False
def check_if_just_year(input):

    if(input.strip()=="XXXX"):
        return True

    split_by_dash=input.split("-")
    if(len(split_by_dash)==1 and split_by_dash[0]=="XXXX"):
        return True

    #2. Dates that include only year information should be removed.
    if (len(split_by_dash) > 1 and split_by_dash[1] == "XX") and split_by_dash[2] == "XX":
        return True

    return False


# replace all years in a given date with a custom value. this is useful because when we calculate the unix date, all dates have same year/grounding
def replace_all_years_with_custom_value(input):
    return input.replace("XXXX", str(CUSTOM_YEAR))



def clean_replace_year(end_date):
    end_date=end_date.strip()
    # if the date is something like "XXXX" , i.e., no month or date, just skip that entry and move on
    if (check_if_just_year(end_date)):
        return False, ""
    else:
        end_date_with_custom_year = replace_all_years_with_custom_value(end_date)
        return True, end_date_with_custom_year

# if its a date range, expand it to include every single date in the range
def get_days_count_in_range(start_date,end_date):
    assert type(start_date) is datetime
    assert type(end_date) is datetime
    assert (end_date-start_date).days >1
    return (end_date-start_date).days


all_dates=[]

def list_all_days_in_a_range(start_date, end_date,time_delta):
    assert type(start_date) is datetime
    assert type(time_delta) is int
    dates_in_range=[]
    for x in range(time_delta):
        dates_in_range.append(datetime.strftime(start_date+timedelta(x),"%Y-%d-%m"))
    dates_in_range.append(datetime.strftime(end_date, "%Y-%d-%m"))
    return dates_in_range

for each_input in INPUTS:
    is_range,range_splits=check_if_range(each_input)
    if(is_range):
        flag1, start_date_with_custom_year = clean_replace_year(range_splits[0])
        flag2, end_date_with_custom_year = clean_replace_year(range_splits[1])
        if flag1  and flag2:
            start_date_as_datetime=datetime.strptime(start_date_with_custom_year, "%Y-%m-%d")
            end_date_as_datetime = datetime.strptime(end_date_with_custom_year, "%Y-%m-%d")
            days_count=get_days_count_in_range(start_date_as_datetime,end_date_as_datetime)
            dates_in_range=list_all_days_in_a_range(start_date_as_datetime,end_date_as_datetime,days_count)
            all_dates.extend(dates_in_range)
        else:
                break
    else: #if its a stand alone date, and not range, clean and add it to the list of all_dates
        flag, end_date_with_custom_year = clean_replace_year(each_input)
        if (flag):
            all_dates.append(end_date_with_custom_year)

print(f"value of alldates is {all_dates}")



