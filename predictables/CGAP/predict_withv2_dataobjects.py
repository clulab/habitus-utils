from sklearn.model_selection import train_test_split
from sklearn import  linear_model,svm,neighbors, tree
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
from v2_Data_Objects import DO_Encoder, DO_Decoder, Encoded_DOs, Decoded_DOs, Decoded_CGAP_DOs
import matplotlib.pyplot as plt
import numpy as np
import sys,os,math
from tqdm import tqdm
from sklearn.model_selection import KFold

#all_countries = ['bgd','cdi','moz','nga','tan','uga']

RUN_ON_SERVER=False
COUNTRY='bgd'
#if you know the survey qn allows for multiple answers from farmer, ensure MULTI_LABEL=True.#todo: do that using code
MULTI_LABEL=False
RANDOM_SEED=3252
FEATURE_SELECTION_ALGOS=["SelectKBest"]
DO_FEATURE_SELECTION=True
MAX_BEST_FEATURE_COUNT='all' #int. else use "all" if you want it to find best k features by combining 1 through all "
NO_OF_BEST_FEATURES_TO_PRINT=20 #even if the best combination has n features print only top 20



DO_NFCV=True #do n fold cross validation instead of train,dev, test splits
N_FOR_NFCV=5 #number of splits for n fold cross validation
NFCV_SELECTKBEST_SPLIT_PERCENTAGE =20 #when using nfcv, split the entire data first into two parts, and use only one part for feature selection

#data related flags:

FILL_NAN_WITH=-1
USE_ALL_DATA=True
# ['COUNTRY', 'Country_Decoded']=housekeeping columns
QNS_TO_AVOID = ['COUNTRY', 'Country_Decoded','D14']
REGEX_QNS_TO_AVOID = ["F+"] #eg:'F+', '' if you dont know the exact qn name, but want to instead avoid training on a subset. eg. all finance related qns,. i.e starting with F
QNS_TO_ADD = ['COUNTRY', 'Country_Decoded','D14',"F1"]
SURVEY_QN_TO_PREDICT= "F58"







random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



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
log_file_name=COUNTRY+"_"+git_details['repo_short_sha']+".log"
log_file_path=os.path.join(os.getcwd(),"logs/",log_file_name)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO ,
    filename=log_file_path,
    filemode='w'
)
logging.getLogger().addHandler(logging.StreamHandler())
CGAP=None
if(RUN_ON_SERVER==True):
    sys.path.append(
        '/work/mithunpaul/habitus/clustering/habitus_clulab_repo/predictables/CGAP/Data/data_objects/code_and_notebooks')
    CGAP = Decoded_CGAP_DOs()
    filepath = '/work/mithunpaul/habitus/clustering/habitus_clulab_repo/predictables/CGAP/Data/data_objects/'
    CGAP.read_and_decode(os.path.join(filepath, 'cgap_json.txt'))
else:
    sys.path.append(
        '/Users/mordor/research/habitus_project/clulab_repo/predictables/CGAP/Data/data_objects/code_and_notebooks')
    CGAP = Decoded_CGAP_DOs()
    filepath = '/Users/mordor/research/habitus_project/clulab_repo/predictables/CGAP/Data/data_objects/'
    CGAP.read_and_decode(os.path.join(filepath, 'cgap_json.txt'))





assert CGAP is not None
if(USE_ALL_DATA==True):
    df_combined = CGAP.get_all_columns_given_country(QNS_TO_AVOID,COUNTRY,REGEX_QNS_TO_AVOID,SURVEY_QN_TO_PREDICT)
else:
    df1=country_name.concat_all_single_answer_qns_to_add(QNS_TO_ADD)
    df2=country_name.concat_all_multiple_answer_qns_to_add(QNS_TO_ADD)
    df_combined = pd.concat([df1, df2], axis=1)

def find_majority_baseline_binary(data, column_name):
    row_count=data[column_name].shape[0]
    yays=(data.loc[data[column_name] == 1]).shape[0]
    nays=row_count-yays
    if yays>nays:
        return ("yes",yays*100/row_count)

    else:
        return ("no", nays * 100 / row_count)

def find_majority_baseline_binary_given_binary_column(column):
    row_count=len(column)
    yays=column.sum()
    nays=row_count-yays
    if yays>nays:
        return ("yes",yays*100/row_count)
    else:
        return ("no",nays*100/row_count)


if not MULTI_LABEL==True:
    maj_class,baseline=find_majority_baseline_binary(df_combined, SURVEY_QN_TO_PREDICT)
    logger.info(f"majority baseline={baseline}, majority class={maj_class} for all_data")
#drop rows which has all values as na
df_combined=df_combined.dropna(how='all')

#if a farmer's reply to the intended qn to predict is nan, then drop that farmer.
cols_qn_to_predict = df_combined.filter(regex=(SURVEY_QN_TO_PREDICT + "_*")).columns
df_combined = df_combined.dropna(how='all', subset=cols_qn_to_predict)



#fill the rest of all nan with some value you pick
df_combined = df_combined.fillna(FILL_NAN_WITH)

# if (MAX_BEST_FEATURE_COUNT)=="ALL": #use all features for feature selection. else use the int value as ciel
#     MAX_BEST_FEATURE_COUNT=df_combined.shape[1]

if not MULTI_LABEL==True:
    maj_class,baseline=find_majority_baseline_binary(df_combined, SURVEY_QN_TO_PREDICT)
    logger.info(f"majority baseline={baseline}, majority class={maj_class} all data without nan")


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


def get_topn_best_feature_names(selectK,data,n):
    features_scores = selectK.scores_
    features_indices = [x for x in range(0, len(features_scores))]
    zipped = zip(features_scores, features_indices)
    sorted_zipped = sorted(zipped,reverse=True)
    topk_best_feature_names=[]
    for (score,index) in sorted_zipped[:n]:
        topk_best_feature_names.append(data.columns[index])
    return topk_best_feature_names


def select_kbest_feature_indices(feature_count, x_train, y_train):
    selectK = SelectKBest(mutual_info_classif, k=feature_count)
    selectK.fit(x_train, y_train)
    selectMask = selectK.get_support()
    best_feature_indices = np.where(selectMask)[0].tolist()
    return best_feature_indices,selectK

def find_best_feature_count():
    best_feature_indices, selectK = select_kbest_feature_indices(MAX_BEST_FEATURE_COUNT, x_train, y_train)
    x_train_selected = x_train.iloc[:, best_feature_indices]
    x_dev_selected = x_dev.iloc[:, best_feature_indices]
    x_train_selected = np.asarray(x_train_selected)
    x_dev_selected = np.asarray(x_dev_selected)
    y_train_gold_selected = np.asarray(y_train)
    y_dev_gold_selected = np.asarray(y_dev)
    model.fit(x_train_selected, y_train_gold_selected)
    y_dev_pred = model.predict(x_dev_selected)
    acc = accuracy_score(y_dev_gold_selected, y_dev_pred)
    list_features.append(MAX_BEST_FEATURE_COUNT)
    list_accuracy.append(acc)
    accuracy_per_feature_count[MAX_BEST_FEATURE_COUNT] = acc
    if (acc > best_feature_accuracy):
        selecK_best = selectK
        best_feature_accuracy = acc
        best_feature_count = MAX_BEST_FEATURE_COUNT
    # plot a figure with number of features as x axis and accuracy as y axis.
    # fig, ax = plt.subplots()
    assert len(list_features) == len(list_accuracy)
    logger.info(f"list_features={list_features}")
    logger.info(f"list_accuracy={list_accuracy}")
    # ax.plot(list_features, list_accuracy)
    # plt.show()


def do_training_predict_given_train_dev_splits_with_feature_selection(model, train, dev, test):
    # separate out the gold/qn to predict so that we train only on the rest
    # we are doing this splitting after train, dev, test split, because we want the data points for predicting qn also to be ssplit/shuffled along with everyone else
    if MULTI_LABEL == True:
        y_train = train.filter(regex=(SURVEY_QN_TO_PREDICT + "_*"))
        x_train = train.drop(y_train.columns, axis=1)
        y_dev = dev.filter(regex=(SURVEY_QN_TO_PREDICT + "_*"))
        x_dev = dev.drop(y_dev.columns, axis=1)
    else:
        y_train = (train[SURVEY_QN_TO_PREDICT])
        x_train = train.drop(SURVEY_QN_TO_PREDICT, axis=1)
        y_dev = np.asarray(dev[SURVEY_QN_TO_PREDICT])
        x_dev = dev.drop(SURVEY_QN_TO_PREDICT, axis=1)

    if not MULTI_LABEL == True:
        maj_class_dev, baseline_dev = find_majority_baseline_binary(dev, SURVEY_QN_TO_PREDICT)
        maj_class_train, baseline_train = find_majority_baseline_binary(train, SURVEY_QN_TO_PREDICT)
        logger.info(f"majority baseline in dev={baseline_dev}, majority class={maj_class_dev}")
        logger.info(f"majority baseline in train={baseline_train}, majority class={maj_class_train}")

    topn = None
    best_feature_accuracy = 0
    best_feature_count = 0
    final_best_combination_of_features = {}
    accuracy_per_feature_count = {}
    selecK_best = None
    if (DO_FEATURE_SELECTION == True):
        feature_accuracy = {}
        list_features = []
        list_accuracy = []
        #for feature_count in tqdm(range(1, MAX_BEST_FEATURE_COUNT), desc="best features", total=MAX_BEST_FEATURE_COUNT):
        best_feature_indices,selectK=select_kbest_feature_indices(MAX_BEST_FEATURE_COUNT,x_train,y_train)
        # selectK = SelectKBest(mutual_info_classif, k=feature_count)
        # selectK.fit(x_train, y_train)
        # selectMask = selectK.get_support()
        # best_feature_indices = np.where(selectMask)[0].tolist()
        x_train_selected = x_train.iloc[:, best_feature_indices]
        x_dev_selected = x_dev.iloc[:, best_feature_indices]
        x_train_selected = np.asarray(x_train_selected)
        x_dev_selected = np.asarray(x_dev_selected)
        y_train_gold_selected = np.asarray(y_train)
        y_dev_gold_selected = np.asarray(y_dev)
        model.fit(x_train_selected, y_train_gold_selected)
        y_dev_pred = model.predict(x_dev_selected)
        acc = accuracy_score(y_dev_gold_selected, y_dev_pred)
        list_features.append(MAX_BEST_FEATURE_COUNT)
        list_accuracy.append(acc)
        accuracy_per_feature_count[MAX_BEST_FEATURE_COUNT] = acc
        if (acc > best_feature_accuracy):
            selecK_best = selectK
            best_feature_accuracy = acc
            best_feature_count = MAX_BEST_FEATURE_COUNT
        # plot a figure with number of features as x axis and accuracy as y axis.
        # fig, ax = plt.subplots()
        assert len(list_features) == len(list_accuracy)
        logger.info(f"list_features={list_features}")
        logger.info(f"list_accuracy={list_accuracy}")
        # ax.plot(list_features, list_accuracy)
        # plt.show()

        logger.debug("\n")
        logger.debug(
            f"****Classification Report when using {type(model).__name__}*** for COUNTRY={COUNTRY} and question to predict={SURVEY_QN_TO_PREDICT} ")
        logger.debug(classification_report(y_dev_gold_selected, y_dev_pred))
        logger.debug("\n")
        logger.debug("****Confusion Matrix***")
        labels_in = [0, 1]
        logger.debug(f"yes\tno")

        cm = confusion_matrix(y_dev_gold_selected, y_dev_pred, labels=labels_in)
        logger.debug(cm)
        logger.debug("\n")
        logger.debug("****True Positive etc***")
        logger.debug('(tn, fp, fn, tp)')
        logger.debug(cm.ravel())

        assert selecK_best is not None
        if (MAX_BEST_FEATURE_COUNT < NO_OF_BEST_FEATURES_TO_PRINT):
            topn = get_topn_best_feature_names(selecK_best, x_train, MAX_BEST_FEATURE_COUNT)
        else:
            topn = get_topn_best_feature_names(selecK_best, x_train, NO_OF_BEST_FEATURES_TO_PRINT)
    else:
        x_train_selected = np.asarray(x_train)
        x_dev_selected = np.asarray(x_dev)
        y_train_gold_selected = np.asarray(y_train)
        y_dev_gold_selected = np.asarray(y_dev)
        if (MULTI_LABEL == True):
            model = MultiOutputClassifier(model).fit(x_train_selected, y_train_gold_selected)
            y_dev_pred = model.predict(x_dev)
            all_acc = np.zeros(y_dev_pred.shape[0])
            multilabelFeature_accuracy = {}
            for index, each_pred_column in enumerate(y_dev_pred.T):
                # find majority class baseline for dev
                maj_class, maj_class_baseline = find_majority_baseline_binary_given_binary_column(
                    y_dev_gold_selected.T[index])
                acc = accuracy_score(y_dev_gold_selected.T[index], each_pred_column)
                column_name = y_dev.columns[index]
                multilabelFeature_accuracy[column_name] = (acc, maj_class_baseline, maj_class)
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
            labels_in = [0, 1]
            logger.debug(f"yes\tno")

            cm = confusion_matrix(y_dev_gold_selected, y_dev_pred, labels=labels_in)
            logger.debug(cm)
            logger.debug("\n")
            logger.debug("****True Positive etc***")
            logger.debug('(tn, fp, fn, tp)')
            logger.debug(cm.ravel())
            acc = accuracy_score(y_dev_gold_selected, y_dev_pred)

    if (DO_FEATURE_SELECTION == True):
        assert topn is not None
        topn_str = ",".join(topn)
        logger.info(
            f"best_feature_count: {str(best_feature_count)} \nbest_feature_accuracy: {str(best_feature_accuracy)} \ntop best features:{topn_str}")
    else:
        if (MULTI_LABEL == True):
            all_accuracies = []
            header1_formatted = "{:<20}".format("Feature Column")
            header2_formatted = "{:<20}".format("Accuracy")
            header3_formatted = "{:<20}".format("Majority Baseline")
            header4_formatted = "{:<20}".format("Majority Class")
            logger.info(f"{header1_formatted}\t{header2_formatted}\t{header3_formatted}\t{header4_formatted}\t")
            for k, v in (multilabelFeature_accuracy.items()):
                k_formatted = "{:<20}".format(str(k))
                v0_formatted = "{:<20}".format(str(round(v[0], 2)))
                v1_formatted = "{:<20}".format(str(round(v[1], 2)))
                v2_formatted = "{:<20}".format(str((v[2])))
                logger.info(f"{k_formatted}\t{v0_formatted}\t{v1_formatted}\t{v2_formatted}")
                all_accuracies.append(v)
            # logger.debug(f"average of all columns={np.mean(all_accuracies)}")


def do_training_predict_given_train_dev_splits_without_feature_selection(model, train, dev, test,best_feature_indices):
    # separate out the gold/qn to predict so that we train only on the rest
    if MULTI_LABEL == True:
        y_train = train.filter(regex=(SURVEY_QN_TO_PREDICT + "_*"))
        x_train = train.drop(y_train.columns, axis=1)
        y_dev = dev.filter(regex=(SURVEY_QN_TO_PREDICT + "_*"))
        x_dev = dev.drop(y_dev.columns, axis=1)
    else:
        y_train = (train[SURVEY_QN_TO_PREDICT])
        x_train = train.drop(SURVEY_QN_TO_PREDICT, axis=1)
        y_dev = np.asarray(dev[SURVEY_QN_TO_PREDICT])
        x_dev = dev.drop(SURVEY_QN_TO_PREDICT, axis=1)

    #apply the mask found from selectkbest onto the training and test slices- do this after taking out SURVEY_QN_TO_PREDICT
    x_train = x_train.iloc[:, best_feature_indices]
    x_dev = x_dev.iloc[:, best_feature_indices]

    x_train_selected = np.asarray(x_train)
    x_dev_selected = np.asarray(x_dev)
    y_train_gold_selected = np.asarray(y_train)
    y_dev_gold_selected = np.asarray(y_dev)
    if (MULTI_LABEL == True):
        model = MultiOutputClassifier(model).fit(x_train_selected, y_train_gold_selected)
        y_dev_pred = model.predict(x_dev)
        all_acc = np.zeros(y_dev_pred.shape[0])
        multilabelFeature_accuracy = {}
        for index, each_pred_column in enumerate(y_dev_pred.T):
            # find majority class baseline for dev
            maj_class, maj_class_baseline = find_majority_baseline_binary_given_binary_column(
                y_dev_gold_selected.T[index])
            acc = accuracy_score(y_dev_gold_selected.T[index], each_pred_column)
            column_name = y_dev.columns[index]
            multilabelFeature_accuracy[column_name] = (acc, maj_class_baseline, maj_class)
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
        labels_in = [0, 1]
        logger.debug(f"yes\tno")

        cm = confusion_matrix(y_dev_gold_selected, y_dev_pred, labels=labels_in)
        logger.debug(cm)
        logger.debug("\n")
        logger.debug("****True Positive etc***")
        logger.debug('(tn, fp, fn, tp)')
        logger.debug(cm.ravel())
        acc = accuracy_score(y_dev_gold_selected, y_dev_pred)
    return acc

train= test =dev = None
if (DO_NFCV == True):
    #split out entire data into two parts, use one part for select k best features.
    selectksplit_datapoint_count=int(df_combined.shape[0]*NFCV_SELECTKBEST_SPLIT_PERCENTAGE/100)
    data_for_selectkbest = df_combined.iloc[:selectksplit_datapoint_count]
    y_data_for_selectkbest = data_for_selectkbest[SURVEY_QN_TO_PREDICT]
    x_data_for_selectkbest = data_for_selectkbest.drop(SURVEY_QN_TO_PREDICT, axis=1)
    best_feature_indices, selectK = select_kbest_feature_indices(MAX_BEST_FEATURE_COUNT, x_data_for_selectkbest,
                                                                 y_data_for_selectkbest)

    best_feature_names=get_topn_best_feature_names(selectK,data_for_selectkbest,20)
    data_not_used_for_selectkbest = df_combined.iloc[selectksplit_datapoint_count:]
    kf = KFold(n_splits=N_FOR_NFCV)
    kf.get_n_splits(data_not_used_for_selectkbest)
    logger.info(
        f"best_feature_count: {str(len(best_feature_indices))}"        
        f"\ntop best features:{best_feature_names}")

    merge_n_test_folds=pd.DataFrame()

    for index,(train_index,test_index) in enumerate(kf.split(data_not_used_for_selectkbest)):
        logger.info(f"*****************starting new fold. Fold number {index+1}")
        train=data_not_used_for_selectkbest.iloc[train_index]
        #also attach back the part you used for seleckbest. note, use only in training. not in dev
        train=pd.concat([train,data_for_selectkbest],axis=0)
        dev=data_not_used_for_selectkbest.iloc[test_index] #in nfcv world there is only train test. but here using the word dev for maintaining consistency with non nfcv world

        #collecting all test folds to use later for majority basecalculation across
        merge_n_test_folds = pd.concat([dev, merge_n_test_folds], axis=0)

        maj_class, baseline = find_majority_baseline_binary(dev, SURVEY_QN_TO_PREDICT)
        #logger.info(f"majority baseline={round(baseline,2)}, majority class={maj_class} dev data for this fold")


        accuracy=do_training_predict_given_train_dev_splits_without_feature_selection(model, train, dev, test,best_feature_indices)
        logger.info(
            f"nbest_feature_accuracy: {str(round(accuracy*100,2))}")

    maj_class, baseline = find_majority_baseline_binary(merge_n_test_folds, SURVEY_QN_TO_PREDICT)
    logger.info(f" for all n folds combined majority baseline={round(baseline,2)}, majority class={maj_class}:")

else:
    train, test_dev = train_test_split(df_combined, test_size=0.2, shuffle=True)
    test, dev = train_test_split(test_dev, test_size=0.5, shuffle=True)

    assert train is not None
    assert test is not None
    assert dev is not None

    do_training_predict_given_train_dev_splits_with_feature_selection(model, train, dev, test)









