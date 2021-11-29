from numpy import random
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-colorblind')
from datetime import datetime
from datetime import timedelta
import numpy
import time


#INPUTS_BEFORE_ERROR_ANALYSIS=["XXXX-07-07 -- XXXX-07-22","XXXX-07-19 -- XXXX-08-04","XXXX-08-01","XXXX-09-25 -- XXXX-10-05","XXXX-05-15","XXXX-07-19 -- XXXX-08-04","1358-XX-XX","XXXX-03-03 -- XXXX-03-11","XXXX-07-14 -- XXXX-07-31","2005-03-18","XXXX-06-17","XXXX-10-25 -- XXXX-12-10","XXXX-10-17","XXXX-10-17","XXXX-03-11 -- XXXX-03-31","2011-XX-XX -- 2018-XX-XX","XXXX-07-07 -- XXXX-07-22","XXXX-07-19 -- XXXX-08-04","2018-05-10","2018-06-21","XXXX-07-19 -- XXXX-08-04","XXXX-06-17","XXXX-12-15 -- XXXX-01-13","XXXX-12-16 -- XXXX-01-15","XXXX-05-05"]
#INPUTS_AFTER_ERROR_ANALYSIS
INPUTS=["XXXX-07-07 -- XXXX-07-22","XXXX-07-19 -- XXXX-08-04","XXXX-07-19 -- XXXX-08-04","XXXX-03-03 -- XXXX-03-11","XXXX-07-14 -- XXXX-07-31","2005-03-18"]
#dates_with_year=INPUT.replace("XXXX", "2017")
#_DATE_RANGE=tuple(dates_with_year.split("--"))


_DATE_FORMAT = '%Y-%m-%d'
_DATE_FORMAT_ONLY_MMDD = '%m-%d'
_EMPIRICAL_SCALE_RATIO = 0.15
_DISTRIBUTION_SIZE = 1000
_SCALE_RATIO_FOR_DRAWING=1e8
GENERATE_DUMMY_DATA=True
CUSTOM_YEAR=1982
_BIN_SIZE=5
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

def check_if_datetime_convertible(input):
    try:
        input_datetime=datetime.strptime(input, _DATE_FORMAT)
        return True
    except ValueError:
        return False
    except OverflowError :
        return False



# replace all years in a given date with a custom value. this is useful because when we calculate the unix date, all dates have same year/grounding
def replace_all_years_with_custom_value(input):
    return input.replace("XXXX", str(CUSTOM_YEAR))



def clean_replace_year(input):
    dat_input=datetime.strptime(input,_DATE_FORMAT)
    dat_input=dat_input.replace(CUSTOM_YEAR,dat_input.month,dat_input.day)
    return True,datetime.strftime(dat_input,_DATE_FORMAT)


def clean_replace_year_str(end_date):
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
    #if the date is spilling into next year, the difference will be negative- return 365-diff
    if ((end_date-start_date).days)>0:
        return (end_date-start_date).days
    else:
        return 365+((end_date-start_date).days)


def convert_list_to_unix_timestamp_list(input):
    list_timestamp=[]
    for x in input:
        list_timestamp.append(datetime.timestamp(x))
    return list_timestamp

def convert_list_str_to_datetime(input):
    list_datetime=[]
    for x in input:
        list_datetime.append(datetime.strptime(x,"%Y-%m-%d"))
    return list_datetime

all_dates_str=[]

def list_all_days_in_a_range(start_date, end_date,time_delta):
    assert type(start_date) is datetime
    assert type(time_delta) is int
    dates_in_range=[]
    for x in range(time_delta):
        dates_in_range.append(datetime.strftime(start_date+timedelta(x),"%Y-%m-%d"))
    dates_in_range.append(datetime.strftime(end_date, "%Y-%m-%d"))
    return dates_in_range

for index,each_input in enumerate(INPUTS):
    is_range,range_splits=check_if_range(each_input)
    if(is_range):
        flag1, start_date_with_custom_year = clean_replace_year_str(range_splits[0])
        flag2, end_date_with_custom_year = clean_replace_year_str(range_splits[1])
        if flag1  and flag2:
            start_date_as_datetime=datetime.strptime(start_date_with_custom_year, "%Y-%m-%d")
            end_date_as_datetime = datetime.strptime(end_date_with_custom_year, "%Y-%m-%d")
            days_count=get_days_count_in_range(start_date_as_datetime,end_date_as_datetime)
            dates_in_range=list_all_days_in_a_range(start_date_as_datetime,end_date_as_datetime,days_count)
            all_dates_str.extend(dates_in_range)
        else:
            continue
    else: #if its a stand alone date, and not range, clean and add it to the list of all_dates
        is_dt_convertible=check_if_datetime_convertible(each_input)
        flag=False
        if(not is_dt_convertible):
            flag, end_date_with_custom_year = clean_replace_year_str(each_input)
        else:
            flag,end_date_with_custom_year=clean_replace_year(each_input)
        if (flag):
            all_dates_str.append(end_date_with_custom_year)

all_dates_datetime_format=convert_list_str_to_datetime(all_dates_str)
all_dates_as_timestamp=convert_list_to_unix_timestamp_list(all_dates_datetime_format)

print(f"value of alldates is {all_dates_as_timestamp}")


distribution=numpy.array(all_dates_as_timestamp)
distribution=np.true_divide(distribution, _SCALE_RATIO_FOR_DRAWING)

assert distribution is not None
sigma=3
mu=1193084540.5079513



fig, axes = plt.subplots(sharex='all',sharey='all',figsize=(13, 7))



count, bins, ignored = axes.hist(distribution, bins=_BIN_SIZE, density=True)


#for printing axes purposes
def reverse_dates(date,pos=None):
    indiv=time.strftime(_DATE_FORMAT_ONLY_MMDD, time.localtime(date*_SCALE_RATIO_FOR_DRAWING))
    return indiv

#to plot the actual distribution for a given sigma and mu i.e when using dummy data
#axes.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

labels = axes.get_xticklabels()

plt.setp(labels, rotation=45, horizontalalignment='right')

title_custom="Fitting a kernel density estimation over a multimodal \ndistribution of sowing dates in a given range\n (lines=bandwidth of kernel density function)"
axes.set(xlim=[bins[0], bins[len(bins)-1]], xlabel='sowing dates', ylabel='No of dates per bin ',
       title=title_custom)

axes.xaxis.set_major_formatter(reverse_dates)

all_bdw=(np.linspace(0.01,0.03,10)).round(2)
for bndw in all_bdw:
    kde=KernelDensity(kernel='gaussian',bandwidth=bndw).fit(X=distribution.reshape(-1, 1))
    log_density=kde.score_samples(bins.reshape(-1,1))
    axes.plot(bins,np.exp(log_density),label=bndw)
axes.legend()
plt.show()



