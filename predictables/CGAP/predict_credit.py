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
import git,logging


COUNTRY='bgd'
GOLD="F58"
RANDOM_SEED=3252
TOTAL_FEATURE_COUNT=27
FEATURE_SELECTION_ALGOS=["SelectKBest"]


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
log_file_name=git_details['repo_short_sha']+".log"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG ,
    filename=log_file_name,
    filemode='w'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))






sys.path.append('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/Code and Notebooks')
Data = CGAP_Decoded()
Data.read_and_decode('/Users/mordor/research/habitus_project/mycode/predictables/Data/Data Objects/CGAP_JSON.txt')

x=Data.bgd_A1

bgd = Country_Decoded(COUNTRY,Data)

#get all data for the given country. then split it into train, dev, split
qns_to_avoid=['COUNTRY','Country_Decoded']
df1=bgd.concat_all_single_answer_qns(qns_to_avoid)
df2=bgd.concat_all_multiple_answer_qns(qns_to_avoid)
assert len(df1)==len(df2)
df_combined = pd.concat([df1, df2], axis=1)
df_combined=df_combined.fillna(-1)

#
# qns_to_add=['F53','F54','F55','F56','F58']
# df1=bgd.concat_all_single_answer_qns_to_add(qns_to_add)
# df2=bgd.concat_all_multiple_answer_qns_to_add(qns_to_add)
# df_combined = pd.concat([df1, df2], axis=1)
# df_combined=df_combined.fillna(9999)




train,test_dev=train_test_split(df_combined,  test_size=0.2,shuffle=True)
test,dev=train_test_split(test_dev,  test_size=0.5,shuffle=True)



y_train_gold=np.asarray(train[GOLD]).reshape(-1, 1)
train.drop(GOLD,inplace=True,axis=1)





y_dev_gold=np.asarray(dev[GOLD])
dev.drop(GOLD,inplace=True,axis=1)


feature_accuracy={}
for feature_count in range(1, TOTAL_FEATURE_COUNT):
    # x_train_selected = SelectKBest(mutual_info_classif, k=feature_count).fit_transform(train, y_train_gold)
    # x_dev_selected = SelectKBest(mutual_info_classif, k=feature_count).fit_transform(dev, y_dev_gold)

    x_train_selected = SelectPercentile(mutual_info_classif, percentile=feature_count).fit_transform(train, y_train_gold)
    x_dev_selected = SelectPercentile(mutual_info_classif, percentile=feature_count).fit_transform(dev, y_dev_gold)

    x_dev_selected=np.asarray(x_dev_selected)
    x_train_selected=np.asarray(x_train_selected)

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


    model = svm.SVC()
    #rfe.fit(X, y)
    # Train the model using the training sets
    model.fit(x_train_selected, y_train_gold)
    y_dev_pred = model.predict(x_dev_selected)

    logger.debug("\n")
    logger.debug(f"****Classification Report when using {type(model).__name__}*** for COUNTRY={COUNTRY} and question to predict={GOLD}")
    logger.debug(classification_report(y_dev_gold, y_dev_pred))
    logger.debug("\n")
    logger.debug("****Confusion Matrix***")
    labels_in=[0,1]
    logger.debug(f"yes\tno")

    cm=confusion_matrix(y_dev_gold, y_dev_pred,labels=labels_in)
    logger.debug(cm)
    logger.debug("\n")
    logger.debug("****True Positive etc***")
    logger.debug('(tn, fp, fn, tp)')
    logger.debug(cm.ravel())


    acc=accuracy_score(y_dev_gold, y_dev_pred)
    feature_accuracy[feature_count]=acc
for k,v in (feature_accuracy.items()):
    logger.info(f"{k}\t{v}")
