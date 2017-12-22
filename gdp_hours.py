from datetime import datetime as dt
import logging as lg
import numpy as np
import pandas as pd
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

LOGGING_LEVEL = lg.DEBUG
YEARS = ('10', '11', '12', '13', '14') # year 2010 to year 2014
TESTYEARS = ('11', '12', '13', '14') # year 2011 to year 2014
IRRELEVANT_IMPACTING_CONDITIONS = ('EQUIPMENT / OUTAGE', 'OTHER / EMERGENCY', 'OTHER / OTHER', 'OTHER / SECURITY', 'RWY-TAXI / CONSTRUCTION', 'RWY-TAXI / DISABLED AIRCRAFT', 'RWY-TAXI / MAINTENANCE', 'VOLUME / COMPACTED DEMAND', 'VOLUME / VOLUME')
COLNAMES = ('TS_hr', 'PC_hr', 'Vis_hr', 'Ceiling_hr', 'CW0422_hr', 'CW1129_hr')
FEATURES = ('Delta_TS', 'Delta_PC', 'Delta_VS', 'Delta_CL', 'Delta_CW0422', 'Delta_CW1129')
assert len(COLNAMES) == len(FEATURES)
FIVE_CLASSES = ('C', 'E', 'R', 'A', 'U')
THREE_CLASSES = ('+', '-', '0')
#CLASSES = FIVE_CLASSES
CLASSES = THREE_CLASSES
SCORER = make_scorer(f1_score, labels=list(CLASSES), average='macro')
DTC_GRID_PARAMS = {}
# Default: DTC_GRID_PARAMS['criterion'] = ['gini']
# Default: DTC_GRID_PARAMS['splitter'] = ['best']
DTC_GRID_PARAMS['max_depth'] = [1, 2, 3, 4, 5]
DTC_GRID_PARAMS['min_samples_split'] = [0.04, 0.08, 0.12, 0.16]
DTC_GRID_PARAMS['min_samples_leaf'] = [0.02, 0.04, 0.06, 0.08]
DTC_GRID_PARAMS['min_weight_fraction_leaf'] = [0.0]
DTC_GRID_PARAMS['max_features'] = [1, 2, 3, 4, 5]
DTC_GRID_PARAMS['random_state'] = np.random.random_integers(0, sys.maxint, 10)
DTC_GRID_PARAMS['max_leaf_nodes'] = [10]
DTC_GRID_PARAMS['min_impurity_decrease'] = [0.0001, 0.0003, 0.001, 0.003, 0.01]
DTC_GRID_PARAMS['class_weight'] = ['balanced']

def load_gdp_hours():
    df = pd.read_csv('GDP advisory hours_V2.csv', delimiter=',', quotechar='"')
    lg.info('CSV Type: ' + str(type(df)) + ', Shape: ' + str(df.shape))
    logger_input.debug('CSV Type: ' + str(type(df)) + ', Shape: ' + str(df.shape))
    num_rows = df.shape[0]
    curr_root_adv_num = -1
    curr_adv_num = -1
    curr_impacting_condition = ''
    curr_adv_dur = -1
    Xyear = {}
    yyear = {}
    for yr in YEARS:
        Xyear[yr] = []
        yyear[yr] = []
    gdp_irrelevant_count = 0
    gdp_orphan = []
    gdp_start = []
    gdp_cancel = []
    gdp_extend = []
    gdp_reduce = []
    gdp_amend = []
    gdp_unchg = []
    for i, row in df[0:num_rows].iterrows():
        logger_input.debug('  Row#: ' + str(i+1) + ', Type: ' + str(type(row)) + ', Length: ' + str(row.shape[0]))
        if (row['RootAdvisoryNumber'] != curr_root_adv_num):
            curr_root_adv_num = row['RootAdvisoryNumber']
            curr_adv_num = row['AdvisoryNumber']
            curr_impacting_condition = row['Impacting.Condition']
            if (curr_impacting_condition in IRRELEVANT_IMPACTING_CONDITIONS):
                logger_input.debug('    Irrelevant impacting condition ' + curr_impacting_condition)
                gdp_irrelevant_count += 1
                continue
            curr_adv_dur = row['Duration_Initiative']
            curr_fs = []
            for col in COLNAMES:
                curr_fs.append(row[col])
            curr_fs = np.array(curr_fs)
            if (row['AdvisoryType'] == 'GDP CNX'):
                gdp_orphan.append(i)
                logger_input.debug('    (O) Orphaned GDP Advisory:     TS ' + str(curr_fs[0]) + ', PC ' + str(curr_fs[1]) + ', VS ' + str(curr_fs[2]) + ', CL ' + str(curr_fs[3]) + ', CW0422 ' + str(curr_fs[4]) + ', CW1129 ' + str(curr_fs[5]))
            else:
                gdp_start.append(i)
                logger_input.debug('    (S) Started New GDP Advisory:  TS ' + str(curr_fs[0]) + ', PC ' + str(curr_fs[1]) + ', VS ' + str(curr_fs[2]) + ', CL ' + str(curr_fs[3]) + ', CW0422 ' + str(curr_fs[4]) + ', CW1129 ' + str(curr_fs[5]))
        elif (row['AdvisoryNumber'] != curr_adv_num): # row['RootAdvisoryNumber'] == curr_root_adv_num
            curr_adv_num = row['AdvisoryNumber']
            if (curr_impacting_condition in IRRELEVANT_IMPACTING_CONDITIONS):
                logger_input.debug('    Irrelevant impacting condition ' + curr_impacting_condition)
                gdp_irrelevant_count += 1
                continue
            prev_adv_dur = curr_adv_dur
            curr_adv_dur = row['Duration_Initiative']
            chg_adv_dur = curr_adv_dur - prev_adv_dur
            prev_fs = curr_fs
            curr_fs = []
            for col in COLNAMES:
                curr_fs.append(row[col])
            curr_fs = np.array(curr_fs)
            chg_fs = curr_fs - prev_fs
            adv_yr = row['AdvisoryDate.UTC'][6:8]
            Xyear[adv_yr].append(chg_fs)
            if (row['AdvisoryType'] == 'GDP CNX'):
                yyear[adv_yr].append('C')
                gdp_cancel.append(i)
                logger_input.debug('    (C) GDP Advisory Cancellation: TS ' + str(curr_fs[0]) + ', PC ' + str(curr_fs[1]) + ', VS ' + str(curr_fs[2]) + ', CL ' + str(curr_fs[3]) + ', CW0422 ' + str(curr_fs[4]) + ', CW1129 ' + str(curr_fs[5]))
                logger_input.debug('    Changes:                       TS ' + str(chg_fs[0]) + ', PC ' + str(chg_fs[1]) + ', VS ' + str(chg_fs[2]) + ', CL ' + str(chg_fs[3]) + ', CW0422 ' + str(chg_fs[4]) + ', CW1129 ' + str(chg_fs[5]))
            elif (chg_adv_dur > 0):
                yyear[adv_yr].append('E')
                gdp_extend.append(i)
                logger_input.debug('    (E) GDP Advisory Extension:    TS ' + str(curr_fs[0]) + ', PC ' + str(curr_fs[1]) + ', VS ' + str(curr_fs[2]) + ', CL ' + str(curr_fs[3]) + ', CW0422 ' + str(curr_fs[4]) + ', CW1129 ' + str(curr_fs[5]))
                logger_input.debug('    Changes:                       TS ' + str(chg_fs[0]) + ', PC ' + str(chg_fs[1]) + ', VS ' + str(chg_fs[2]) + ', CL ' + str(chg_fs[3]) + ', CW0422 ' + str(chg_fs[4]) + ', CW1129 ' + str(chg_fs[5]))
            elif (chg_adv_dur < 0):
                yyear[adv_yr].append('R')
                gdp_reduce.append(i)
                logger_input.debug('    (R) GDP Advisory Reduction:    TS ' + str(curr_fs[0]) + ', PC ' + str(curr_fs[1]) + ', VS ' + str(curr_fs[2]) + ', CL ' + str(curr_fs[3]) + ', CW0422 ' + str(curr_fs[4]) + ', CW1129 ' + str(curr_fs[5]))
                logger_input.debug('    Changes:                       TS ' + str(chg_fs[0]) + ', PC ' + str(chg_fs[1]) + ', VS ' + str(chg_fs[2]) + ', CL ' + str(chg_fs[3]) + ', CW0422 ' + str(chg_fs[4]) + ', CW1129 ' + str(chg_fs[5]))
            else:
                yyear[adv_yr].append('A')
                gdp_amend.append(i)
                logger_input.debug('    (A) GDP Advisory Amendment:    TS ' + str(curr_fs[0]) + ', PC ' + str(curr_fs[1]) + ', VS ' + str(curr_fs[2]) + ', CL ' + str(curr_fs[3]) + ', CW0422 ' + str(curr_fs[4]) + ', CW1129 ' + str(curr_fs[5]))
                logger_input.debug('    Changes:                       TS ' + str(chg_fs[0]) + ', PC ' + str(chg_fs[1]) + ', VS ' + str(chg_fs[2]) + ', CL ' + str(chg_fs[3]) + ', CW0422 ' + str(chg_fs[4]) + ', CW1129 ' + str(chg_fs[5]))
        else: # row['RootAdvisoryNumber'] == curr_root_adv_num, row['AdvisoryNumber'] == curr_adv_num
            if (curr_impacting_condition in IRRELEVANT_IMPACTING_CONDITIONS):
                logger_input.debug('    Irrelevant impacting condition ' + curr_impacting_condition)
                gdp_irrelevant_count += 1
                continue
            prev_fs = curr_fs
            curr_fs = []
            for col in COLNAMES:
                curr_fs.append(row[col])
            curr_fs = np.array(curr_fs)
            chg_fs = curr_fs - prev_fs
            adv_yr = row['AdvisoryDate.UTC'][6:8]
            Xyear[adv_yr].append(chg_fs)
            yyear[adv_yr].append('U')
            gdp_unchg.append(i)
            logger_input.debug('    (U) GDP Advisory Unchanged:    TS ' + str(curr_fs[0]) + ', PC ' + str(curr_fs[1]) + ', VS ' + str(curr_fs[2]) + ', CL ' + str(curr_fs[3]) + ', CW0422 ' + str(curr_fs[4]) + ', CW1129 ' + str(curr_fs[5]))
            logger_input.debug('    Changes:                       TS ' + str(chg_fs[0]) + ', PC ' + str(chg_fs[1]) + ', VS ' + str(chg_fs[2]) + ', CL ' + str(chg_fs[3]) + ', CW0422 ' + str(chg_fs[4]) + ', CW1129 ' + str(chg_fs[5]))
    lg.info('  Number of irrelevant advisories  : ' + str(gdp_irrelevant_count))
    lg.info('  Number of orphaned advisories (O): ' + str(len(gdp_orphan)))
    lg.info('  Number of started advisories  (S): ' + str(len(gdp_start)))
    lg.info('  Total number of advisories  (O+S): ' + str(len(gdp_orphan) + len(gdp_start)))
    lg.info('  Number of cancellations       (C): ' + str(len(gdp_cancel)))
    lg.info('  Number of extensions          (E): ' + str(len(gdp_extend)))
    lg.info('  Number of reductions          (R): ' + str(len(gdp_reduce)))
    lg.info('  Number of amendments          (A): ' + str(len(gdp_amend)))
    lg.info('  Total number of changes (C+E+R+A): ' + str(len(gdp_cancel) + len(gdp_extend) + len(gdp_reduce) + len(gdp_amend)))
    lg.info('  Number of unchangeds          (U): ' + str(len(gdp_unchg)))
    lg.info('  Total number of hours (C+E+R+A+U): ' + str(len(gdp_cancel) + len(gdp_extend) + len(gdp_reduce) + len(gdp_amend) + len(gdp_unchg)))
    for yr in YEARS:
        Xyear[yr] = np.array(Xyear[yr])
        yyear[yr] = np.array(yyear[yr])
        Xyear[yr] = Xyear[yr].reshape(-1, len(FEATURES))
    return Xyear, yyear

def split_dataset(Xyear, yyear):
    # Year 2011 will contain train = {2010}, test = {2011}
    # Year 2012 will contain train = {2010, 2011}, test = {2012}
    # Year 2013 will contain train = {2010, 2011, 2012}, test = {2013}
    # Year 2014 will contain train = {2010, 2011, 2012, 2013}, test = {2014}
    for yr in YEARS:
        lg.info('Year 20' + str(yr))
        lg.info('  Number of {C, E, R, A, U} samples: ' + str((yyear[yr] == 'C').sum()) + ', ' + str((yyear[yr] == 'E').sum()) + ', ' + str((yyear[yr] == 'R').sum()) + ', ' + str((yyear[yr] == 'A').sum()) + ', ' + str((yyear[yr] == 'U').sum()))
        lg.info('  Total number of samples:           ' + str(yyear[yr].shape[0]))
    Xtrain = {}
    ytrain = {}
    Xtest = {}
    ytest = {}
    for tt in TESTYEARS:
        Xtrain[tt] = []
        ytrain[tt] = []
        for yr in YEARS:
            if (int(yr) < int(tt)):
                Xtrain[tt].extend(Xyear[yr])
                ytrain[tt].extend(yyear[yr])
        Xtrain[tt] = np.array(Xtrain[tt])
        ytrain[tt] = np.array(ytrain[tt])
        Xtest[tt] = np.array(Xyear[tt])
        ytest[tt] = np.array(yyear[tt])
        Xtrain[tt] = Xtrain[tt].reshape(-1, len(FEATURES))
        Xtest[tt] = Xtest[tt].reshape(-1, len(FEATURES))
        lg.info('Train and test sets for year 20' + str(tt))
        lg.info('  Number of {C, E, R, A, U} train samples: ' + str((ytrain[tt] == 'C').sum()) + ', ' + str((ytrain[tt] == 'E').sum()) + ', ' + str((ytrain[tt] == 'R').sum()) + ', ' + str((ytrain[tt] == 'A').sum()) + ', ' + str((ytrain[tt] == 'U').sum()))
        lg.info('  Total number of train samples:           ' + str(ytrain[tt].shape[0]))
        lg.info('  Number of {C, E, R, A, U} test samples:  ' + str((ytest[tt] == 'C').sum()) + ', ' + str((ytest[tt] == 'E').sum()) + ', ' + str((ytest[tt] == 'R').sum()) + ', ' + str((ytest[tt] == 'A').sum()) + ', ' + str((ytest[tt] == 'U').sum()))
        lg.info('  Total number of test samples:            ' + str(ytest[tt].shape[0]))
        if CLASSES == THREE_CLASSES:
            ytrain[tt][ytrain[tt] == 'C'] = '-'
            ytrain[tt][ytrain[tt] == 'E'] = '+'
            ytrain[tt][ytrain[tt] == 'R'] = '-'
            ytrain[tt][ytrain[tt] == 'A'] = '0'
            ytrain[tt][ytrain[tt] == 'U'] = '0'
            ytest[tt][ytest[tt] == 'C'] = '-'
            ytest[tt][ytest[tt] == 'E'] = '+'
            ytest[tt][ytest[tt] == 'R'] = '-'
            ytest[tt][ytest[tt] == 'A'] = '0'
            ytest[tt][ytest[tt] == 'U'] = '0'
            lg.info('Train and test sets for year 20' + str(tt) + ' (merged into 3 classes)')
            lg.info('  Number of {+, -, 0} train samples: ' + str((ytrain[tt] == '+').sum()) + ', ' + str((ytrain[tt] == '-').sum()) + ', ' + str((ytrain[tt] == '0').sum()))
            lg.info('  Total number of train samples:     ' + str(ytrain[tt].shape[0]))
            lg.info('  Number of {+, -, 0} test samples:  ' + str((ytest[tt] == '+').sum()) + ', ' + str((ytest[tt] == '-').sum()) + ', ' + str((ytest[tt] == '0').sum()))
            lg.info('  Total number of test samples:      ' + str(ytest[tt].shape[0]))
    return Xtrain, ytrain, Xtest, ytest

def split_dataset_for_indices(Xyear, yyear):
    # Merge training data into one list
    Xtrainall = []
    ytrainall = []
    for yr in YEARS:
        Xtrainall.extend(Xyear[yr])
        ytrainall.extend(yyear[yr])
    Xtrainall = np.array(Xtrainall)
    ytrainall = np.array(ytrainall)
    if CLASSES == THREE_CLASSES:
        ytrainall[ytrainall == 'C'] = '-'
        ytrainall[ytrainall == 'E'] = '+'
        ytrainall[ytrainall == 'R'] = '-'
        ytrainall[ytrainall == 'A'] = '0'
        ytrainall[ytrainall == 'U'] = '0'
    lg.info('Number of samples in Xtrainall: ' + str(Xtrainall.shape[0]))
    lg.info('Number of samples in ytrainall: ' + str(ytrainall.shape[0]))
    # Create indices lists for custom CV split
    trainindiceslist = []
    testindiceslist = []
    for i in range(len(TESTYEARS)):
        tt = TESTYEARS[i]
        trainindiceslist.append([])
        testindiceslist.append([])
        startindex = 0
        endindex = 0
        for yr in YEARS:
            startindex = endindex
            endindex += yyear[yr].shape[0]
            if (int(yr) < int(tt)):
                trainindiceslist[i].extend(range(startindex, endindex))
            elif (int(yr) == int(tt)):
                testindiceslist[i].extend(range(startindex, endindex))
        lg.info('  Number of samples in trainindiceslist[' + str(i) + ']: ' + str(len(trainindiceslist[i])))
        lg.info('  Number of samples in testindiceslist [' + str(i) + ']: ' + str(len(testindiceslist[i])))
    return Xtrainall, ytrainall, trainindiceslist, testindiceslist

def compute_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    lg.info('  Accuracy        : ' + str(accuracy))
    informedness = recall_score(y_true, y_pred, labels=list(CLASSES), average='macro')
    lg.info('  Macro recall    : ' + str(informedness))
    markedness = precision_score(y_true, y_pred, labels=list(CLASSES), average='macro')
    lg.info('  Macro precision : ' + str(markedness))
    f1score = f1_score(y_true, y_pred, labels=list(CLASSES), average='macro')
    lg.info('  Macro F1 score  : ' + str(f1score))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    lg.info('  Confusion matrix:')
    lg.info(str(conf_matrix))
    return f1score

if __name__ == '__main__':
    # Step 0: Set up logging mechanism
    dt_now = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
    formatter = lg.Formatter(fmt='%(message)s')
    handler_console = lg.StreamHandler()
    handler_console.setLevel(lg.INFO)
    handler_console.setFormatter(formatter)
    handler_info = lg.FileHandler(filename='gdp_log_' + dt_now + '_info.txt', mode='w')
    handler_info.setLevel(lg.INFO)
    handler_info.setFormatter(formatter)
    handler_input = lg.FileHandler(filename='gdp_log_' + dt_now + '_input.txt', mode='w')
    handler_input.setLevel(lg.DEBUG)
    handler_input.setFormatter(formatter)
    handler_output = lg.FileHandler(filename='gdp_log_' + dt_now + '_output.txt', mode='w')
    handler_output.setLevel(lg.DEBUG)
    handler_output.setFormatter(formatter)
    logger = lg.getLogger('')
    logger.setLevel(LOGGING_LEVEL)
    logger.addHandler(handler_console)
    logger.addHandler(handler_info)
    logger_input = lg.getLogger('INPUT')
    logger_input.addHandler(handler_input)
    logger_output = lg.getLogger('OUTPUT')
    logger_output.addHandler(handler_output)
    # Step 1: Load data
    lg.info('--------------------------------------------------')
    lg.info('Step 1: Load data')
    lg.info('--------------------------------------------------')
    Xyear, yyear = load_gdp_hours()
    # Step 2: Group into train and test sets by year
    lg.info('--------------------------------------------------')
    lg.info('Step 2: Group into train and test sets by year')
    lg.info('--------------------------------------------------')
    Xtrain, ytrain, Xtest, ytest = split_dataset(Xyear, yyear)
    # Step 3: Calculate train and test sets indices
    lg.info('--------------------------------------------------')
    lg.info('Step 3: Calculate train and test sets indices')
    lg.info('--------------------------------------------------')
    Xtrainall, ytrainall, trainindiceslist, testindiceslist = split_dataset_for_indices(Xyear, yyear)
    # Step 4: Run Naive Bayes
    lg.info('--------------------------------------------------')
    lg.info('Step 4: Run Naive Bayes')
    lg.info('--------------------------------------------------')
    gnb = GaussianNB()
    scoresgnb = []
    for tt in TESTYEARS:
        # Calculate sample weights for training set
        weights = np.ones(ytrain[tt].shape[0], float)
        weights = weights * ytrain[tt].shape[0] / len(CLASSES)
        for c in CLASSES:
            weights[ytrain[tt] == c] /= (ytrain[tt] == c).sum()
        # Run Naive Bayes
        lg.info('Running Naive Bayes on train and test sets for year 20' + str(tt) + ' and classes ' + str(CLASSES))
        ypred = gnb.fit(Xtrain[tt], ytrain[tt], weights).predict(Xtest[tt])
        logger_output.debug('Running Naive Bayes on train and test sets for year 20' + str(tt) + ' and classes ' + str(CLASSES))
        for i in range(Xtest[tt].shape[0]):
            logger_output.debug('  Row#: ' + str(i+1) + ', X: ' + str(Xtest[tt][i]) + ', y: ' + str(ytest[tt][i]) + ', ypred: ' + str(ypred[i]))
        # Output metrics
        scoresgnb.append(compute_scores(ytest[tt], ypred))
    # Step 5: Find Best Parameters for Decision Tree Classifier using Grid Search
    lg.info('--------------------------------------------------')
    lg.info('Step 5: Find Best Parameters for Decision Tree Classifier using Grid Search')
    lg.info('--------------------------------------------------')
    lg.info('Running Grid Search on Decision Tree Classifier on classes ' + str(CLASSES))
    dtc = DecisionTreeClassifier()
    dtcgs = GridSearchCV(estimator=dtc, param_grid=DTC_GRID_PARAMS, scoring=SCORER, n_jobs=4, refit=True, cv=zip(trainindiceslist, testindiceslist), verbose=1)
    dtcgs.fit(Xtrainall, ytrainall)
    dtcbest = dtcgs.best_estimator_
    lg.info('  Best parameters         : ' + str(dtcgs.best_params_))
    lg.info('  Best score              : ' + str(dtcgs.best_score_))
    lg.info('  Best feature importances: ' + str(dtcbest.feature_importances_))
    # Step 6: Run Decision Tree Classifier using Best Parameters
    lg.info('--------------------------------------------------')
    lg.info('Step 6: Run Decision Tree Classifier using Best Parameters')
    lg.info('--------------------------------------------------')
    scoresdtcnone = []
    scoresdtcsmote = []
    scoresdtctomek = []
    smote = SMOTE(ratio='auto', kind='regular');
    tomek = TomekLinks(ratio='auto');
    for tt in TESTYEARS:
        # Run Decision Tree Classifier
        lg.info('Running Decision Tree Classifier on train and test sets for year 20' + str(tt) + ' and classes ' + str(CLASSES))
        ypred = dtcbest.fit(Xtrain[tt], ytrain[tt]).predict(Xtest[tt])
        logger_output.debug('Running Decision Tree Classifier on train and test sets for year 20' + str(tt) + ' and classes ' + str(CLASSES))
        for i in range(Xtest[tt].shape[0]):
            logger_output.debug('  Row#: ' + str(i+1) + ', X: ' + str(Xtest[tt][i]) + ', y: ' + str(ytest[tt][i]) + ', ypred: ' + str(ypred[i]))
        # Output metrics
        scoresdtcnone.append(compute_scores(ytest[tt], ypred))
        # Run Decision Tree Classifier with SMOTE oversampling
        lg.info('Running Decision Tree Classifier (with SMOTE) on train and test sets for year 20' + str(tt) + ' and classes ' + str(CLASSES))
        Xsmote, ysmote = smote.fit_sample(Xtrain[tt], ytrain[tt])
        ypred = dtcbest.fit(Xsmote, ysmote).predict(Xtest[tt])
        logger_output.debug('Running Decision Tree Classifier (with SMOTE) on train and test sets for year 20' + str(tt) + ' and classes ' + str(CLASSES))
        for i in range(Xtest[tt].shape[0]):
            logger_output.debug('  Row#: ' + str(i+1) + ', X: ' + str(Xtest[tt][i]) + ', y: ' + str(ytest[tt][i]) + ', ypred: ' + str(ypred[i]))
        # Output metrics
        scoresdtcsmote.append(compute_scores(ytest[tt], ypred))
        # Run Decision Tree Classifier with TOMEK undersampling
        lg.info('Running Decision Tree Classifier (with TOMEK) on train and test sets for year 20' + str(tt) + ' and classes ' + str(CLASSES))
        Xtomek, ytomek = tomek.fit_sample(Xtrain[tt], ytrain[tt])
        ypred = dtcbest.fit(Xtomek, ytomek).predict(Xtest[tt])
        logger_output.debug('Running Decision Tree Classifier (with TOMEK) on train and test sets for year 20' + str(tt) + ' and classes ' + str(CLASSES))
        for i in range(Xtest[tt].shape[0]):
            logger_output.debug('  Row#: ' + str(i+1) + ', X: ' + str(Xtest[tt][i]) + ', y: ' + str(ytest[tt][i]) + ', ypred: ' + str(ypred[i]))
        # Output metrics
        scoresdtctomek.append(compute_scores(ytest[tt], ypred))
    # Step 7: Calculate f-statistic
    lg.info('--------------------------------------------------')
    lg.info('Step 7: Calculate f-statistic')
    lg.info('--------------------------------------------------')
    lg.info('Calculating f-statistic for Decision Tree Classifier (No resampling vs SMOTE oversampling vs TOMEK undersampling)')
    fstatistic, fpvalue = f_oneway(scoresdtcnone, scoresdtcsmote, scoresdtctomek)
    lg.info('  F-statistic: ' + str(fstatistic) + ', P-Value: ' + str(fpvalue))
    # Step 8: Calculate t-statistic
    lg.info('--------------------------------------------------')
    lg.info('Step 8: Calculate t-statistic')
    lg.info('--------------------------------------------------')
    lg.info('Calculating t-statistic for Naive Bayes and Decision Tree Classifier (No resampling)')
    tstatistic, tpvalue = ttest_ind(scoresgnb, scoresdtcnone, equal_var=False)
    lg.info('  T-statistic: ' + str(tstatistic) + ', P-Value: ' + str(tpvalue))
    # Step 9: Fit Decision Tree to Whole Dataset and Export to File
    lg.info('--------------------------------------------------')
    lg.info('Step 9: Fit Decision Tree to Whole Dataset and Export to File')
    lg.info('--------------------------------------------------')
    lg.info('Running Decision Tree Classifier on whole dataset')
    dtcbest = dtcbest.fit(Xtrainall, ytrainall)
    dot_data = export_graphviz(dtcbest, out_file=None, feature_names=FEATURES, class_names=CLASSES, filled=True, rounded=True, special_characters=True)
    graph = graph_from_dot_data(dot_data)
    graph.write_png('gdp_tree_' + dt_now + '.png')
    lg.info('  Decision Tree Exported to File:' + 'gdp_tree_' + dt_now + '.png')

