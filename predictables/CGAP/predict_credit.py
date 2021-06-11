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
#if you know the qn is multi-label, ensure MULTI_LABEL=True.
#todo: do that using code
SURVEY_QN_TO_PREDICT= "F53"
MULTI_LABEL=True
RANDOM_SEED=3252
TOTAL_FEATURE_COUNT=2
FEATURE_SELECTION_ALGOS=["SelectKBest"]
FILL_NAN_WITH=-1
DO_FEATURE_SELECTION=False
USE_ALL_DATA=True
#when training using all qns in the survey are there any qns you would want the classifier not to train on.
# e.g., housekeeping columns like Country_Decoded. or if you want to remove handpicked qns , you add them to the list here
QNS_TO_AVOID = ['COUNTRY', 'Country_Decoded']
#QNS_TO_AVOID = ['COUNTRY', 'Country_Decoded','F53','F54','F55','F56']

#a bunch of hand picked qns only one which you want to train. Ensure USE_ALL_DATA=False
QNS_TO_ADD=['F53','F54','F55','F56','F58']


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
os.path.join
log_file_name=os.path.join(os.getcwd(),"logs/",git_details['repo_short_sha']+".log")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG ,
    filename=log_file_name,
    filemode='w'
)
logging.getLogger().addHandler(logging.StreamHandler())

sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')
Data = CGAP_Decoded()
Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')
bgd = Country_Decoded(COUNTRY,Data)


#use only a couple of hand picked qns during training
if(USE_ALL_DATA==True):
    # use all qns from survey in training sans whatever you want to avoid
    df1 = bgd.concat_all_single_answer_qns(QNS_TO_AVOID)
    df2 = bgd.concat_all_multiple_answer_qns(QNS_TO_AVOID)
    assert len(df1) == len(df2)
    df_combined = pd.concat([df1, df2], axis=1)
    df_combined = df_combined.fillna(FILL_NAN_WITH)
else:

    df1=bgd.concat_all_single_answer_qns_to_add(QNS_TO_ADD)
    df2=bgd.concat_all_multiple_answer_qns_to_add(QNS_TO_ADD)
    df_combined = pd.concat([df1, df2], axis=1)
    df_combined=df_combined.fillna(-1)


train,test_dev=train_test_split(df_combined,  test_size=0.2,shuffle=True)
test,dev=train_test_split(test_dev,  test_size=0.5,shuffle=True)


#separate out the gold/qn to predict so that we train only on the rest
if MULTI_LABEL==True:
    y_train_gold=train.filter(regex=(SURVEY_QN_TO_PREDICT+"_*"))
    x_train=train.drop(y_train_gold.columns, axis=1)
    y_dev_gold = dev.filter(regex=(SURVEY_QN_TO_PREDICT+"_*"))
    x_dev=dev.drop(y_dev_gold.columns, axis=1)
else:
    y_train_gold=(train[SURVEY_QN_TO_PREDICT])
    x_train =train.drop(SURVEY_QN_TO_PREDICT,axis=1)
    y_dev_gold=np.asarray(dev[SURVEY_QN_TO_PREDICT])
    x_dev=dev.drop(SURVEY_QN_TO_PREDICT, axis=1)

assert len(x_train.columns) == len(df_combined.columns) - len(y_train_gold.columns)
assert len(x_dev.columns) == len(df_combined.columns) - len(y_dev_gold.columns)

feature_accuracy={}
for feature_count in range(1, TOTAL_FEATURE_COUNT):
    if(DO_FEATURE_SELECTION==True):
        columns_scores = SelectKBest(mutual_info_classif, k=feature_count).fit(x_train, y_train_gold)
        best_feature_indices = np.argpartition(columns_scores.scores_, -feature_count)[-feature_count:]
        best_features = []
        for b in best_feature_indices:
            best_features.append(x_train.columns[b])
        x_train_selected = SelectKBest(mutual_info_classif, k=feature_count).fit_transform(x_train, y_train_gold)
        x_dev_selected = SelectKBest(mutual_info_classif, k=feature_count).fit_transform(x_dev, y_dev_gold)
        # x_train_selected = SelectPercentile(chi2, percentile=feature_count).fit_transform(x_train, y_train_gold)
        # x_dev_selected = SelectPercentile(chi2, percentile=feature_count).fit_transform(x_dev, y_dev_gold)
    x_train=np.asarray(x_train)
    x_dev=np.asarray(x_dev)
    y_train_gold = np.asarray(y_train_gold)
    y_dev_gold = np.asarray(y_dev_gold)

    #MLP
    #model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    #model=neighbors.KNeighborsClassifier()
    #model = LogisticRegression()
    #model = tree.DecisionTreeClassifier()
    #model = RandomForestClassifier(n_estimators=10)
    #model = Perceptron(tol=1e-3, random_state=0)
    #model = svm.SVC()
    #model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    #model = GaussianNB()
    #model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    #model = MLkNN(k=20)
    model=neighbors.KNeighborsClassifier()

    if (MULTI_LABEL==True):
        model=MultiOutputClassifier(model).fit(x_train, y_train_gold)
        y_dev_pred = model.predict(x_dev)
        print(y_dev_pred)
        print(f"shape of y_dev_pred={y_dev_pred.shape}")
        all_acc=np.zeros(y_dev_pred.shape[0])
        for index,each_row in enumerate(y_dev_pred):
            acc = accuracy_score(y_dev_gold[index], each_row)
            all_acc[index]=acc
        print(f"average accuracy across all multi label class predictions={np.mean(all_acc)}")

    else:
        model.fit(x_train, y_train_gold)
        y_dev_pred = model.predict(x_dev)

    # logger.debug("\n")
    # logger.debug(
    #     f"****Classification Report when using {type(model).__name__}*** for COUNTRY={COUNTRY} and question to predict={SURVEY_QN_TO_PREDICT} when using {feature_count} best features")
    # logger.debug(classification_report(y_dev_gold, y_dev_pred))
    # logger.debug("\n")
    # logger.debug("****Confusion Matrix***")
    # labels_in=[0,1]
    # logger.debug(f"yes\tno")
    #
    # cm=confusion_matrix(y_dev_gold, y_dev_pred,labels=labels_in)
    # logger.debug(cm)
    # logger.debug("\n")
    # logger.debug("****True Positive etc***")
    # logger.debug('(tn, fp, fn, tp)')
    # logger.debug(cm.ravel())

    #acc=accuracy_score(y_dev_gold, y_dev_pred)
    if (DO_FEATURE_SELECTION == True):
        feature_accuracy[feature_count]=(str(acc),",".join(best_features))
if(DO_FEATURE_SELECTION==True):
    logger.info("Number of k best features\t accuracy:feature list")
    for k,v in (feature_accuracy.items()):
        logger.info(f"{k}\t{v}")
#else:
    #print(f"accuracy={acc}")


