import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn import  linear_model,svm,neighbors, tree
import matplotlib.pyplot as plt
from CGAP_JSON_Encoders_Decoders import *
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB,CategoricalNB
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif,SelectPercentile
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN
import git,logging,os
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

COUNTRY='bgd'
#if you know the survey qn allows for multiple answers from farmer, ensure MULTI_LABEL=True.#todo: do that using code


RANDOM_SEED=3252
RUN_ON_SERVER=False

FEATURE_SELECTION_ALGOS=["SelectKBest"]
FILL_NAN_WITH=-1

DO_FEATURE_SELECTION=False
USE_ALL_DATA=True
TOTAL_FEATURE_COUNT=680
QNS_TO_AVOID = ['COUNTRY', 'Country_Decoded']
SURVEY_QN_TO_PREDICT="F53"
MULTI_LABEL=True



#Notes:
# ['COUNTRY', 'Country_Decoded']=housekeeping columns


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

all_countries = ['bgd','cdi','moz','nga','tan','uga']

logger = logging.getLogger(__name__)
def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_sha=str(repo.head.object.hexsha),
    repo_short_sha= str(repo.git.rev_parse(repo_sha, short=6))
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
        "repo_short_sha" :repo_short_sha
    }
    return repo_infos

git_details=get_git_info()
log_file_name=os.path.join(os.getcwd(),"logs/",git_details['repo_short_sha']+".log")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG ,
    filename=log_file_name,
    filemode='w'
)
logging.getLogger().addHandler(logging.StreamHandler())

if(RUN_ON_SERVER==True):
    sys.path.append('/work/mithunpaul/habitus/clustering/habitus_clulab_repo/predictables/CGAP/Data/Data Objects/Code and Notebooks')
    Data = CGAP_Decoded()
    Data.read_and_decode('/work/mithunpaul/habitus/clustering/habitus_clulab_repo/predictables/CGAP/Data/Data Objects/CGAP_JSON.txt')
else:
    sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')
    Data = CGAP_Decoded()
    Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')



bgd = Country_Decoded(COUNTRY,Data)


if(USE_ALL_DATA==True):
    df1 = bgd.concat_all_single_answer_qns(QNS_TO_AVOID)
    df2 = bgd.concat_all_multiple_answer_qns(QNS_TO_AVOID)
    assert len(df1) == len(df2)
    df_combined = pd.concat([df1, df2], axis=1)
else:
    df1=bgd.concat_all_single_answer_qns_to_add(QNS_TO_ADD)
    df2=bgd.concat_all_multiple_answer_qns_to_add(QNS_TO_ADD)
    df_combined = pd.concat([df1, df2], axis=1)

def find_majority_baseline_binary(data, column_name):
    row_count=data[column_name].shape[0]
    yays=(data.loc[data[column_name] == 1]).shape[0]
    nays=row_count-yays
    if yays>nays:
        return yays*100/row_count
    else:
        return nays*100/row_count

def find_majority_baseline_binary_given_binary_column(column):
    row_count=len(column)
    yays=column.sum()
    nays=row_count-yays
    if yays>nays:
        return ("yes",yays*100/row_count)
    else:
        return ("no",nays*100/row_count)


if not MULTI_LABEL==True:
    baseline=find_majority_baseline_binary(df_combined, SURVEY_QN_TO_PREDICT)
    logger.info(f"majority baseline={baseline}")
#drop rows which has all values as na
df_combined=df_combined.dropna(how='all')

#if a farmer's reply to the intended qn to predict is nan, then drop that farmer.
cols_qn_to_predict = df_combined.filter(regex=(SURVEY_QN_TO_PREDICT + "_*")).columns
df_combined = df_combined.dropna(how='all', subset=cols_qn_to_predict)

#fill the rest of all nan with some value you pick
df_combined = df_combined.fillna(FILL_NAN_WITH)



train,test_dev=train_test_split(df_combined,  test_size=0.2,shuffle=True)
test,dev=train_test_split(test_dev,  test_size=0.5,shuffle=True)


#separate out the gold/qn to predict so that we train only on the rest
if MULTI_LABEL==True:
    y_train_gold=train.filter(regex=(SURVEY_QN_TO_PREDICT + "_*"))
    x_train=train.drop(y_train_gold.columns, axis=1)
    y_dev_gold = dev.filter(regex=(SURVEY_QN_TO_PREDICT + "_*"))
    x_dev=dev.drop(y_dev_gold.columns, axis=1)
else:
    y_train_gold=(train[SURVEY_QN_TO_PREDICT])
    x_train =train.drop(SURVEY_QN_TO_PREDICT,axis=1)
    y_dev_gold=np.asarray(dev[SURVEY_QN_TO_PREDICT])
    x_dev=dev.drop(SURVEY_QN_TO_PREDICT, axis=1)

#model = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#model=neighbors.KNeighborsClassifier()
#model = LogisticRegression()
#model = tree.DecisionTreeClassifier()
#model = RandomForestClassifier(n_estimators=10)
#model = Perceptron(tol=1e-3, random_state=0)
#model = svm.SVC()
#model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
#model = GaussianNB()
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
#model = MLkNN(k=20)
best_feature_accuracy=0
final_best_combination_of_features={}
if(DO_FEATURE_SELECTION==True):
    feature_accuracy = {}
    for feature_count in range(1, TOTAL_FEATURE_COUNT):
        selectK = SelectKBest(mutual_info_classif, k=feature_count)
        selectK.fit(x_train, y_train_gold)
        selectMask=selectK.get_support()
        best_feature_indices = np.where(selectMask)[0].tolist()
        best_features = []
        #to get the name of the features to print later
        for b in best_feature_indices:
            best_features.append(x_train.columns[b])
        x_train_selected = x_train.iloc[:,best_feature_indices]
        x_dev_selected = x_dev.iloc[:,best_feature_indices]
        # x_train_selected = SelectPercentile(chi2, percentile=feature_count).fit_transform(x_train, y_train_gold)
        # x_dev_selected = SelectPercentile(chi2, percentile=feature_count).fit_transform(x_dev, y_dev_gold)
        x_train_selected=np.asarray(x_train_selected)
        x_dev_selected=np.asarray(x_dev_selected)
        y_train_gold_selected = np.asarray(y_train_gold)
        y_dev_gold_selected = np.asarray(y_dev_gold)
        model.fit(x_train_selected, y_train_gold_selected)
        y_dev_pred = model.predict(x_dev_selected)
        acc = accuracy_score(y_dev_gold_selected, y_dev_pred)
        if(acc>best_feature_accuracy):
            best_feature_accuracy=acc
            final_best_combination_of_features["best_features"] = ("feature_count:",str(feature_count),"\naccuracy:",str(acc),"\nfeatures:", ",".join(best_features))
else:
    x_train_selected = np.asarray(x_train)
    x_dev_selected = np.asarray(x_dev)
    y_train_gold_selected = np.asarray(y_train_gold)
    y_dev_gold_selected = np.asarray(y_dev_gold)





    if (MULTI_LABEL==True):

        model=MultiOutputClassifier(model).fit(x_train_selected, y_train_gold_selected)
        y_dev_pred = model.predict(x_dev)
        all_acc=np.zeros(y_dev_pred.shape[0])
        multilabelFeature_accuracy={}
        for index, each_pred_column in enumerate(y_dev_pred.T):
            # find majority class baseline for dev
            maj_class,maj_class_baseline=find_majority_baseline_binary_given_binary_column(y_dev_gold_selected.T[index])
            acc = accuracy_score(y_dev_gold_selected.T[index], each_pred_column)
            column_name=y_dev_gold.columns[index]
            multilabelFeature_accuracy[column_name]=(acc,maj_class_baseline,maj_class)
            logger.debug("\n")
            logger.debug("**********************************************************************************")
            logger.debug(
                f"****Classification Report when using {type(model).__name__}*** for COUNTRY={COUNTRY} and question to predict={SURVEY_QN_TO_PREDICT} for column name {column_name}")
            logger.debug(f"Majority class={maj_class}; Majority baseline={maj_class_baseline}")
            logger.debug(classification_report(y_dev_gold_selected.T[index], each_pred_column))
            logger.debug("\n")
            logger.debug("****Confusion Matrix***")

            cm = confusion_matrix(y_dev_gold_selected.T[index], each_pred_column)
            logger.debug(cm)
            logger.debug("\n")
            logger.debug("****True Positive etc***")
            logger.debug('(tn, fp, fn, tp)')
            logger.debug(cm.ravel())

    else:
        model.fit(x_train_selected, y_train_gold_selected)
        y_dev_pred = model.predict(x_dev_selected)
        acc = accuracy_score(y_dev_gold_selected, y_dev_pred)

        logger.debug("\n")
        logger.debug(
            f"****Classification Report when using {type(model).__name__}*** for COUNTRY={COUNTRY} and question to predict={SURVEY_QN_TO_PREDICT} ")
        logger.debug(classification_report(y_dev_gold_selected, y_dev_pred))
        logger.debug("\n")
        logger.debug("****Confusion Matrix***")
        labels_in=[0,1]
        logger.debug(f"yes\tno")

        cm=confusion_matrix(y_dev_gold_selected, y_dev_pred, labels=labels_in)
        logger.debug(cm)
        logger.debug("\n")
        logger.debug("****True Positive etc***")
        logger.debug('(tn, fp, fn, tp)')
        logger.debug(cm.ravel())
        acc=accuracy_score(y_dev_gold_selected, y_dev_pred)


if(DO_FEATURE_SELECTION==True):
    logger.info("Number of k best features\t accuracy:feature list")
    logger.info(final_best_combination_of_features)
else:
    if (MULTI_LABEL == True):
        all_accuracies=[]
        logger.info("Feature Column\t\taccuracy\tmajority baseline\tmajority class")
        for k, v in (multilabelFeature_accuracy.items()):
            logger.info(f"{k}\t\t{round(v[0],2)}\t{v[1]}\t{v[2]}")
            all_accuracies.append(v)
        #logger.debug(f"average of all columns={np.mean(all_accuracies)}")


