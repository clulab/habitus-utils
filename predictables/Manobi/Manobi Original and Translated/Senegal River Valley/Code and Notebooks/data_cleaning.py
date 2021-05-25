#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 20:42:38 2021

@author: prcohen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:31:20 2020

@author: prcohen
"""
import sys
sys.path.append('/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/Manobi farmer')
import pyAgrum as gum
import numpy as np
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

filepath = "/Users/prcohen/anaconda2/envs/aPRAM/Habitus/Data/Manobi Data/Original Steven 2017b/"
village = pd.read_excel(filepath+"translated_2017b/village_2017b_translated.xls")
farmer  = pd.read_excel(filepath+"translated_2017b/farmer_2017b_translated.xls")
plot    = pd.read_excel(filepath+"translated_2017b/plot_2017b_translated.xls")
season  = pd.read_excel(filepath+"translated_2017b/agSeason_2017b_translated.xls")

def strip_spaces (df):
    # strip leading and trailing spaces in all string-valued column names and values
    df.columns = [x.strip() for x in df.columns]
    return df.applymap(lambda x: x.strip() if type(x) == str else x)

village = strip_spaces(village)
farmer = strip_spaces(farmer)
plot = strip_spaces(plot)
season = strip_spaces(season)




village_ht = {village.iloc[i]['X.ID']:{'kids': []} for i in range(len(village))}

def make_child_ht(child):
    return {child['X.ID']: {'parent_id': child['Parent.ID'], 'kids': [], 'has_parent':0} for child in
            [child.iloc[i] for i in range(len(child))]}

farmer_ht, plot_ht, season_ht  = make_child_ht(farmer), make_child_ht(plot), make_child_ht(season)

def check_parent_child (child_ht,parent_ht,child_name,parent_name):
    for k,v in child_ht.items():
        parent = parent_ht.get(v['parent_id']) # look for the parent id in parent
        if parent is not None:
            parent['kids'].append(k)            # add the child id to parent's kids
            v['has_parent'] = 1
    kids_with_parents = np.sum([v['has_parent'] for v in child_ht.values()])
    print (f"{kids_with_parents} of {len(child_ht)} {child_name}s exist in {parent_name} data")
    parents_without_kids = np.sum([v['kids']==[] for v in parent_ht.values()])
    print (f"{parents_without_kids} of {len(parent_ht)} {parent_name}s have no {child_name} data")


check_parent_child(farmer_ht,village_ht,'farmer','village')
check_parent_child(plot_ht,farmer_ht,'plot','farmer')
check_parent_child(season_ht,plot_ht,'season','plot')

# missing_villages = []
# for k,v in farmer_ht.items():
#     parent = village_ht.get(v['parent_id']) # look for the parent id in parent
#     if parent is None:
#         missing_villages.append((v['parent_id'],k))



def complete_chain (season_id):
    s = season_ht.get(season_id)
    if s is not None:
        p = plot_ht.get(s['parent_id'])
        if p is not None:
            f = farmer_ht.get(p['parent_id'])
            if f is not None:
                v = village_ht.get(f['parent_id'])
                if v is not None:
                    return True
    return False

print (f"We can trace a season record through plot, farmer and village for {sum([complete_chain(season_id) for season_id in season_ht.keys()])} records")



###############################################################################
########  Clean the village data

village.drop(columns=[
    'fertilizerType_season2',
    'Parent.ID'
    ],
    inplace = True, errors = 'ignore')

village.rename(columns = {
    'X.ID' : 'village_id',
    'villageName\n        Name': 'village_name',
    'Rain gauge in the village': 'has_rain_gauge',
    'villageCenter' : 'lat_long'
    },
    inplace = True)


###############################################################################
########  Clean the farmer data

farmer.drop(columns=[
    'Status', 'Other', 'AnnEe.d.adhEsion.','Search for buyer',
    'paymentDelay', 'Search for buyer.1','Search for buyer.2',
    'Search for buyer.3',
    'othercrop_season1\n  storageLocation2\n If other, precise',
    'If yes, amount to cover\n  If yes, ability to pay',
    'Si.oui..montant..garantir.','Unnamed'
    ],
    inplace = True, errors = 'ignore')

farmer.rename(columns = {
    'X.ID' : 'id',
    'Parent.ID':'village_id',
    'Object.state' : 'validation_status',
    'Accuracy' : 'accuracy',
    'farmerCooperative' : 'coop_name',
    'Access to electricity':'access_electricity',
    'Access to drinking water' : 'access_drinking_water',
    'Access to health care' : 'access_healthcare',
    'Housing material' : 'housing_material',
    'numberWives' : 'num_wives',
    'wifeTrades' : 'wife_trades',
    'wifeFoodCrops' : 'wife_food_crops',
    'wifeCrafts' : 'wife_crafts',
    'childrenWorking' : 'children_working',
    'Transportation means' : 'transportation',
    'wantsCredit': 'wants_credit',
    'creditCooperative' : 'credit_from_coop',
    'creditThirdParty' : 'credit_from_third_party',
    'creditBank' : 'credit_from_bank',
    'loanAmount' : 'loan_amount',
    'loanRate' : 'loan_rate',
    'existingBridgeLoan' : 'has_bridge_loan',
    'bridgeLoanAmount' : 'bridge_loan_amount',
    'wantsBridgeLoan' : 'wants_bridge_loan',
    'wantsBridgeLoanAmount' : 'wants_bridge_loan_amount',
    'pledgedOutput' : 'pledged_output',
    'inputUse' : 'input_use',
    'inputCosts' : 'input_cost',
    'lastPrice' : 'last_price',
    'totalHa' : 'total_ha',
    'mainCrop' : 'main_crop',
    'secondaryCrop' : 'secondary_crop',
    'Nbr children < 6 years' : 'num_young_children',
    'Nbr children 6-18 years' : 'num_older_children',
    'numChildren' : 'num_children',
    'numChildrenInSchool' : 'num_children_in_school',
    'educationLevel' : 'education_level',
    'numPlots_farmer' : 'num_plots',
    'Nbr years in activity' : 'years_farming',
    'Inputs usefulness' : 'inputs_useful',
    'IntErIt.usage.intrants' : 'interest_use_inputs',
    'IntErIt.assurance.agricole' : 'interest_insurance',
    'Agricultural services' : 'use_ag_services',
    'Farm advisory quality' : 'ag_services_quality',
    'Farm advisory frequency' : 'ag_services_frequency',
    'Interest for farm advisory' : 'interest_ag_services',
    'buyerType' : 'buyer_type'
                       },
    inplace = True)


##############################################################################
### correct the ranges of farmer. For example, no-one is 200 years old.

def correct_range (df, col, minval, maxval, report=False):
    n = df[col].count()
    if report: print(f"Before cleaning, {col} count = {df[col].count()}, min = {df[col].min()}, max = {df[col].max()}")
    df[col].where((df[col]>=minval) & (df[col]<=maxval),inplace=True)
    if report: print(f"{df[col].describe()}")
    if report: print(f"Newly Nan records: {n - df[col].count()}\n")

#correct_range('totalHa',0,20,report=True)
correct_range(farmer,'age',18,100)
correct_range(farmer,'years_farming',0,100)
correct_range(farmer,'num_children',0,20)
correct_range(farmer,'income',0,2000000)
correct_range(farmer,'has_bridge_loan', 0, 1000000)
correct_range(farmer,'wants_bridge_loan_amount',0,1500000)
correct_range(farmer,'input_cost',0,500000)
correct_range(farmer,'last_price', 0, 400)
correct_range(farmer,'num_plots', 0, 10)

# Total Ha is very unreliable. ~ 350 farms have total_ha > 3, which is pretty big for
# smallhold farms.  The mean size of these farms is 34.27 Ha, and in several villages
# all the farms have a single value for total_ha.  For example, village 708 has dozens of farms
# each of which has 70Ha as total_ha. I'm setting the total_ha value to nan for every farm
# in each of these villages because these values are constant and large for these villages.

for i in range(len(farmer)):
    farm = farmer.iloc[i]
    if farm.village_id in [448, 490, 495, 500, 503, 567, 674, 675, 681, 684, 682,
                           708, 715, 747, 751, 754]:
        farmer.at[i,'total_ha'] = np.nan

# and for the few very large total_ha values that are not in these villages, I will
# correct the range

correct_range(farmer,'total_ha', 0, 10)

#it appears that farmer seems to code "no loan" as loanAmount = 0 and loanRate = 0
# print(pd.crosstab(farmer.loanAmount>0,farmer.loanRate>0))
# So I will create a new column called have_loan with values 0 and 1 where I code
# the off-diagonal values as nan

def loan_status (row):
    if row.loan_amount == 0 and row.loan_rate == 0:
        return 0
    elif row.loan_amount > 0 and row.loan_rate > 0:
        return 1
    else:
        return np.nan

farmer['has_loan'] = farmer.apply(loan_status,axis=1)

# Now we can correct the ranges on loanAmount and loanStatus
correct_range(farmer,'loan_amount', 1, 1000000)
correct_range(farmer,'loan_rate',.1,20)

farmer['valid_children'] = farmer.num_children >= farmer.num_children_in_school
farmer['fraction_in_school'] = [
    s/n if (v and n > 0) else np.nan for s,n,v in zip(farmer.num_children_in_school, farmer.num_children, farmer.valid_children) ]

###############################################################################
### Recode some variables

def hash_to_integers (col):
    """ Builds a hash table that maps the values in col to integers starting at 0.
    There will be as many unique integers as there are unique values. Returns the
    hash table."""
    ht, i = {}, -1
    for val in col:
        if ht.get(val) is None:
            i+=1
            ht[val] = i
    return ht

# use xxhash package

farmer['coop_id'] = farmer.coop_name
farmer.coop_id.replace(hash_to_integers(farmer.coop_name),inplace=True)

farmer.access_electricity.replace(
    to_replace={"not at all":0, "Compteur": 1, "Solaire":1},inplace=True)

farmer.access_drinking_water.replace(
    to_replace={"RiviEre/Marigot":0, "Puit": 1, "Pompe hydraulique":1, "Robinet":2},inplace=True)

farmer.access_healthcare.replace(
    to_replace={"not at all":0, "bad":0, "medium": 1, "good":2, "very good":2},inplace=True)

farmer.housing_material.replace(
    to_replace={"Paille/Plastique": 0, "Bois":0, "Terre sEchEe/Banco": 1, "Briques":2, "BEton":2},inplace=True)

farmer.gender.replace(
    to_replace={"Mr": 0, "Mme":1, "Mlle":2},inplace=True)

farmer.wife_trades.replace(    to_replace={"No":0,"Yes":1}, inplace=True)
farmer.wife_food_crops.replace( to_replace={"No":0,"Yes":1}, inplace=True)
farmer.wife_crafts.replace(    to_replace={"No":0,"Yes":1}, inplace=True)

farmer.children_working.replace(
    to_replace = {'never':0, 'sometimes': 1, 'during vacation only': 2,
                  'Plusieurs fois par semaine': 3, 'tous les jours': 4},
    inplace=True)

farmer.wants_credit.replace(
    to_replace= {'not interested' : 0, 'a little interested' : 1,
                 'moderately interested' : 2, 'interested' : 3, 'very interested' : 4},
    inplace=True)

farmer.credit_from_coop.replace(
    to_replace={'never' : 0, 'sometimes' : 1, 'always': 2},
    inplace=True)

farmer.credit_from_third_party.replace(
    to_replace={'never' : 0, 'sometimes' : 1, 'always': 2},
    inplace=True)

farmer.credit_from_bank.replace(
    to_replace={'never' : 0, 'sometimes' : 1, 'always': 2},
    inplace=True)

farmer.pledged_output.replace(
    to_replace={'no' : 0, 'partly' : 1, 'yes': 2},
    inplace=True)

farmer.wants_bridge_loan.replace(to_replace={'no' : 0, 'yes' : 1}, inplace=True)

farmer.interest_insurance.replace(
    to_replace={'no' : 0, 'maybe' : 1, 'Peut-Itre': 1, 'yes' : 2},
    inplace=True)

farmer.input_use.replace(
    to_replace={'not at all' : 0, 'very little' : 1, 'sometimes' : 2,
                'half recommended dose' : 3, 'recommended dose' : 4},
    inplace=True)

farmer.inputs_useful.replace(
    to_replace={'no' : 0, 'maybe' : 1, 'Peut-Itre': 1, 'yes' : 2},
    inplace=True)

farmer.interest_use_inputs.replace(
    to_replace={'no' : 0, 'maybe' : 1, 'Peut-Itre': 1, 'yes' : 2},
    inplace=True)

farmer.use_ag_services.replace(
    to_replace={'Aucun' : 0, 'ANCAR' : 2, 'Autres' : 1},
    inplace=True)

farmer.ag_services_quality.replace(
    to_replace={'not at all' : 0, 'fair': 0, 'medium' : 1, 'good':2, 'very good':3},
    inplace=True)

farmer.ag_services_frequency.replace(
    to_replace={
        'Aucune visite' : 0,
        'any visite' : 0,
        '1  2 visites par an': 1,
        '1  2 visites par semestres' : 2,
        '1  2 visites par trimestres' : 2,
        '1 visite par mois' : 3,
        '2 visites par mois' : 4},
    inplace=True)

farmer.interest_ag_services.replace(
    to_replace={'no' : 0, 'maybe' : 1, 'Peut-Itre': 1, 'yes' : 2},
    inplace=True)

farmer.buyer_type.replace(
    to_replace={'market' : 'market', 'lender': 'lender',
                'CoopErative':'coop','itinerant merchant' : 'merchant'},
    inplace=True)

farmer.main_crop.replace(
    to_replace={'irrigated rice': 'IRice','rainfed rice': 'RRice',
                'market gardening': 'garden'},
    inplace=True)

farmer.secondary_crop.replace(
    to_replace={'irrigated rice': 'IRice',
                'rainfed rice': 'RRice',
                'market gardening': 'garden',
                'Elevage': 'elevage',
                'maize': 'maize',
                'NiEbE': 'niebe',
                'Sorgho':'sorghum',
                'millet' : 'millet',
                'SEsame':'sesame',
                'peanut': 'peanut'},
    inplace=True)


###############################################################################
### Binning some continuous variables

def map_to_bins (attribute,bins):
    farmer[attribute+'_binned'] = pd.cut(
        farmer[attribute],
        bins,
        labels = list(range(len(bins)-1)),
        right=False).astype(float)
    # astype(float) otherwise the bin nums are treated as categories and we can't take their mean etc.

def map_to_quartile_bins (attribute):
    fa = farmer[attribute]
    bins = [fa.min(),fa.quantile(.25),fa.quantile(.5),fa.quantile(.75),fa.max()+.1]
    map_to_bins (attribute,bins)

map_to_quartile_bins('fraction_in_school')
map_to_quartile_bins('income')

fis = farmer.fraction_in_school
bins = [fis.min(),fis.quantile(.25),fis.quantile(.5),fis.quantile(.75),fis.max()+.1]
map_to_bins('fraction_in_school',bins)


map_to_bins ('num_plots', [0,2,4,farmer.num_plots.max()+1])
map_to_bins ('years_farming',[0,10,18,28,farmer['years_farming'].max()+1])
map_to_bins ('age', [0,30,40,50,60,farmer.age.max()+1])
map_to_bins ('num_children', [0,4,8,farmer.num_children.max()+1])
map_to_bins('num_older_children',[0,1,5,999])
map_to_bins('num_young_children',[0,1,5,999])
map_to_bins('num_wives',[0,1,2,farmer.num_wives.max()+1]) # 0 for 0 wives, 1 for 1 wife, 2 for >1 wife

def education (row):
    """returns 0 if illiterate or literate, 1 is education level is elementary school,
    middle school or high-school and above"""
    if row.education_level == 'AlphabEtisE' or row.education_level == 'Non AlphabEtisE':
        return 0
    elif  row.education_level == 'Primaire' or row.education_level == 'SupErieur' or row.education_level == 'Secondaire':
        return 1
    else:
        return np.nan

farmer['education_level_binned'] = farmer.apply(education,axis=1)

def women_role (row):
    if row.num_wives > 0:
        if row.wife_trades == 0 and row.wife_food_crops == 0:
            return 0
        if (row.wife_trades == 1 or row.wife_food_crops == 1 ) and (not (row.wife_trades == 1 and row.wife_food_crops == 1 )):
            return 1
        if (row.wife_trades == 1 and row.wife_food_crops == 1 ):
            return 2
    else:
        return np.nan

farmer['wife_roles'] = farmer.apply(women_role,axis=1)


def creditworthy (row):
    if row.has_loan == 0 and row.credit_from_bank == 0:
        return 0
    elif row.has_loan == 1:
        return 1
    else:
        return np.nan

farmer['creditworthy'] = farmer.apply(creditworthy,axis=1)

def positive_income (row):
    if row.income > 0:
        return row.income
    else:
        return np.nan

farmer['nonzero_income'] = farmer.apply(positive_income,axis=1)


## add a list of plot ids to each farmer

farmer['plot_list'] = [[] for i in range(len(farmer))]
parent_to_plot_ht = {}
for k,v in plot_ht.items():
    p = v['parent_id']
    if parent_to_plot_ht.get(p):
        parent_to_plot_ht[p].append(k)
    else:
        parent_to_plot_ht[p] = [k]

farmer['plot_list'] = farmer.apply(lambda x: parent_to_plot_ht.get(x.id) or np.nan, axis=1)

farmer.to_csv(filepath+"processed/demographicfarmer_2017b_cleaned.csv",index=False)

#%%

def N (x):
    return np.sum(~np.isnan(x))

def PCT (x):
    return np.sum(~np.isnan(x))/len(x)

farmer_attributes = [
    'gender', 'age', 'years_farming',
    'num_wives', 'wife_trades', 'wife_food_crops', 'wife_crafts',
    'num_children', 'num_young_children', 'num_older_children',
    'num_children_in_school', 'children_working',
    'housing_material', 'access_electricity', 'access_drinking_water', 'access_healthcare',
    'wants_credit','credit_from_coop', 'credit_from_third_party', 'credit_from_bank',
    'has_loan', 'loan_amount', 'loan_rate', 'has_bridge_loan', 'bridge_loan_amount',
    'pledged_output', 'wants_bridge_loan', 'wants_bridge_loan_amount',
    'input_use', 'input_cost', 'inputs_useful', 'interest_use_inputs',
    'interest_insurance', 'use_ag_services', 'ag_services_quality',
    'ag_services_frequency', 'interest_ag_services',
    'last_price', 'num_plots', 'total_ha',
    'fraction_in_school', 'fraction_in_school_binned',
    'num_older_children_binned', 'num_young_children_binned',
    'income', 'nonzero_income', 'income_binned', 'num_plots_binned',
    'years_farming_binned', 'age_binned', 'num_children_binned', 'num_wives_binned',
    'education_level_binned', 'wife_roles','creditworthy'
    ]

####  Make a table of village statistics

g = farmer.groupby('village_id')
village.index = village.village_id  # to ensure that the following concat aligns by village
all_stats = g[farmer_attributes].agg([np.nanmean,np.nanmedian,N])
all_stats.columns = ["_".join(x) for x in all_stats.columns.ravel()]

z = pd.concat([village,g.size(),all_stats],axis=1)

z.rename(columns = {0:'village_size'}, inplace = True)

z.to_excel(filepath+"processed/village_stats_20210209.xlsx",index=False)

#######################  Make a table of coop statistics  #######################

# first make a groupby table of coop names and sizes indexed by coop ids
def coop_name (col):
    return list(set(col))[0]

id_and_name = farmer[['coop_id','coop_name']].groupby('coop_id',as_index=True).agg(coop_name)

# Note that because some farmers do not belong to coops, they have Nan as the coop name:
id_and_name[pd.isna(id_and_name.coop_name)]

# rather than delete this row, losing data about farmers who aren't in coops, we'll
# just replace the NaN
id_and_name.coop_name.fillna('not_in_a_coop',inplace=True)

# then calculate all the stats
g = farmer.groupby('coop_id',as_index=True)
all_stats = g[farmer_attributes].agg([np.nanmean,np.nanmedian,N])
all_stats.columns = ["_".join(x) for x in all_stats.columns.ravel()]

# then concatenate these with the coop sizes

z = pd.concat([id_and_name,g.size(),all_stats],axis=1)
z.rename(columns = {0:'coop_size'}, inplace = True)
z.to_excel(filepath+"processed/coop_stats_20210209.xlsx",index=False)

