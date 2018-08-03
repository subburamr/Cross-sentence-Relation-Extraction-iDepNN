from __future__ import print_function
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
import os
import re
import itertools as it
import numpy as np
import load_save_pkl
from getMacroFScore import getMacroFScore, getIndex, getRelation
from sklearn.externals import joblib

# Different versions - svm_type:
#     1. full sentence represented as bow
#     2. full sentence + entity seg
#     3. full sentence + entity seg + POS tags + e2 type
#     4. full sentence + entity seg + POS tags + dependency
#     5. full sentence + entity seg + POS tags + dependency present in the shortest path
#     6. SDP + entity seg + POS tags + e2 type
#     7. SDP + entity seg + POS tags + dependency present in the shortest path
#     8. full sentence + SDP + entity seg + POS tags + e2 type

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def prepare_bow_corpus(train_data, dev_data, test_data):
    corpus = set()
    datasets = [train_data, dev_data, test_data]
    for dataset in datasets:
        for i, line in enumerate(dataset):
            training_example_list = line[0][0].strip().split("::")
            sent = training_example_list[5]
            entity_tags = ['<e1>','</e1>','<e2>','</e2>']
            for tag in entity_tags:
                sent = sent.replace(tag,"")
            sent_tokenize_list = sent_tokenize(sent)
            for sentence in sent_tokenize_list:
                tokenized_sent = word_tokenize(sentence)
                word_token_list_lower = [word.lower() for word in tokenized_sent]
                corpus.update(word_token_list_lower)
    return corpus

def prepare_svm_data(data, svm_type, resources_path):
    training_data_list_sdp = []
    training_data_list_full = []
    final_training_data_list = []
    training_label_list = []
    training_doc_ids = []
    entity_seg_list = []
    pos_tag_list = []
    dependency_tag_list = []
    sdp_dependency_tag_list = []
    dataset_sdp_sent_list = []


    corpus = load_save_pkl.load_pickle_file(os.path.join(resources_path, "muc6_bow_corpus.pkl"))
    corpus_dict ={word:i for i, word in enumerate(corpus)}
    vectorizer = CountVectorizer(vocabulary=corpus_dict)

    if svm_type == 3 or svm_type == 4 or svm_type == 5 or svm_type == 6 or svm_type == 7 or svm_type == 8:
        pos_corpus = load_save_pkl.load_pickle_file(resources_path + "pos_corpus.pkl")
        pos_corpus_dict = {word:i for i, word in enumerate(pos_corpus)}
        pos_vectorizer = CountVectorizer(lowercase=False, vocabulary=pos_corpus_dict)

    if svm_type==4 or svm_type == 5 or svm_type == 7:
        dependency_corpus = load_save_pkl.load_pickle_file(resources_path + "dependency_corpus.pkl")
        dependency_corpus_dict ={word:i for i, word in enumerate(dependency_corpus)}
        dependency_vectorizer = CountVectorizer(lowercase=False, vocabulary=dependency_corpus_dict)

    for i, line in enumerate(data):
        word_token_list = set()
        training_example_list = line[0][0].strip().split("::")
        doc_id = training_example_list[0]
        rel = training_example_list[2]
        e1 = training_example_list[3]
        e2 = training_example_list[4]

        if svm_type == 1 or svm_type == 2 or svm_type == 3 or svm_type == 4 or svm_type == 5 or svm_type == 8:
            sent = training_example_list[5]
            entity_tags = ['<e1>','</e1>','<e2>','</e2>']
            for tag in entity_tags:
                sent = sent.replace(tag,"")
            sent_tokenize_list = sent_tokenize(sent)
            for sentence in sent_tokenize_list:
                tokenized_sent = word_tokenize(sentence)
                word_token_list_lower = [word.lower() for word in tokenized_sent]
                word_token_list.update(word_token_list_lower)

            training_data_list_full.append(csr_matrix.toarray(vectorizer.transform(word_token_list)).sum(axis=0))

        if svm_type == 6 or svm_type == 7 or svm_type == 8:
            sdp_sent = []
            data_sdp = line[1][1]
            for path_tuple in data_sdp:
                path_dep = path_tuple[1]
                sdp_sent.append(path_dep[1])
            dataset_sdp_sent_list.append(sdp_sent)
            training_data_list_sdp.append(csr_matrix.toarray(vectorizer.transform(sdp_sent)).sum(axis=0))

        training_label_list.append(np.asarray(re.findall('\d+', rel), dtype=np.int)[0])
        training_doc_ids.append(doc_id)

        if svm_type!=1:
            entity_seg_list.append(csr_matrix.toarray(vectorizer.transform([e1.lower(), e2.lower()])).sum(axis=0))

        if svm_type == 3 or svm_type==4 or svm_type == 5:
            pos_tagged_sent_list = []
            for sentence in sent_tokenize_list:
                tokenized_sent = word_tokenize(sentence)
                pos_tagged_sent = nltk.pos_tag(tokenized_sent)
                pos_tagged_sent_list.extend([pair[1] for pair in pos_tagged_sent])
            pos_tag_list.append(csr_matrix.toarray(pos_vectorizer.transform(pos_tagged_sent_list)).sum(axis=0))

    if svm_type == 6 or svm_type==7 or svm_type == 8:
        pos_tagged_sent_list = []
        for sdp_sent in dataset_sdp_sent_list:
            pos_tagged_sent = nltk.pos_tag(sdp_sent)
            pos_tagged_sent_list = [pair[1] for pair in pos_tagged_sent]
            pos_tag_list.append(csr_matrix.toarray(pos_vectorizer.transform(pos_tagged_sent_list)).sum(axis=0))

    if svm_type == 4:
        dependency_tagged_sent_list = []
        for i, line in enumerate(data):
            data_dep = line[1][0]
            # print(data_dep)
            for sent_dep in data_dep:
                for token_dep in sent_dep:
                    dependency_tagged_sent_list.append(token_dep[-1])
            dependency_tag_list.append(csr_matrix.toarray(dependency_vectorizer.transform(dependency_tagged_sent_list)).sum(axis=0))

    if svm_type == 5 or svm_type == 7:
        sdp_dependency_tagged_sent_list = []
        for i, line in enumerate(data):
            data_sdp = line[1][1]
            for path_tuple in data_sdp:
                path_dep_list = path_tuple[1]
                sdp_dependency_tagged_sent_list.append(path_dep_list[-1])
            sdp_dependency_tag_list.append(csr_matrix.toarray(dependency_vectorizer.transform(sdp_dependency_tagged_sent_list)).sum(axis=0))

    if svm_type == 1:
        training_data_list = training_data_list_full

    if svm_type !=1:
        if svm_type == 6 or svm_type == 7 or svm_type == 8:
            training_data_list_sent = training_data_list_sdp
        if svm_type == 2 or svm_type == 3 or svm_type == 4 or svm_type == 5:
            training_data_list_sent = training_data_list_full
        training_data_list_new = [np.append(training_data,entity_seg) for training_data, entity_seg in zip(training_data_list_sent, entity_seg_list)]
        training_data_list = training_data_list_new

    if svm_type==3 or svm_type==4 or svm_type == 5 or svm_type == 6 or svm_type == 7 or svm_type == 8:
        training_data_list_new = [np.append(training_data,pos_tag) for training_data, pos_tag in zip(training_data_list, pos_tag_list)]
        training_data_list = training_data_list_new

    if svm_type == 4:
        training_data_list_new = [np.append(training_data,dependency_tag) for training_data, dependency_tag in zip(training_data_list, dependency_tag_list)]
        training_data_list = training_data_list_new

    if svm_type == 5 or svm_type == 7:
        training_data_list_new = [np.append(training_data,sdp_dependency_tag) for training_data, sdp_dependency_tag in zip(training_data_list, sdp_dependency_tag_list)]
        training_data_list = training_data_list_new

    if svm_type == 8:
        training_data_list_new = [np.append(training_data_full,training_data) for training_data_full, training_data in zip(training_data_list_full, training_data_list)]
        training_data_list = training_data_list_new

    return zip(training_data_list, training_label_list, training_doc_ids)


if __name__ == '__main__':
    experiment_name = "muc6"
    resources_path = '../../../../data/resources/'
    data_path = '../../../../data/processed_input/MUC6/'

    train_k_val = 'k_le_1'
    svm_type = 3
    test_k_vals = ['k_0', 'k_le_1', 'k_le_2', 'k_le_3']

    for test_k_val in test_k_vals:
        results_path = './data/results/results_' + str(experiment_name) + '_train_' + train_k_val + '_test_' + test_k_val + '/'

        train_data_path = data_path + 'train/' + train_k_val + '/'
        dev_data_path = data_path + 'dev/' + train_k_val + '/'
        test_data_path = data_path + 'test/' + test_k_val + '/'

        ensure_directory(results_path)

        train_data = load_save_pkl.load_pickle_file(os.path.join(train_data_path, experiment_name +
                                                                 '_train_data_' + train_k_val + '_sdp_dep.pkl'))
        dev_data = load_save_pkl.load_pickle_file(os.path.join(dev_data_path, experiment_name +
                                                               '_dev_data_' + train_k_val + '_sdp_dep.pkl'))
        test_data = load_save_pkl.load_pickle_file(os.path.join(test_data_path, experiment_name +
                                                                '_test_data_' + test_k_val + '_sdp_dep.pkl'))
        svm_train_data = prepare_svm_data(train_data, svm_type, resources_path)
        svm_dev_data = prepare_svm_data(dev_data, svm_type, resources_path)
        svm_test_data = prepare_svm_data(test_data, svm_type, resources_path)
    
        variants = {
            "kernel" :['linear', 'rbf'],
            "C" : [1, '10', ],
            "gamma" : [1e-3],
        }

        varNames = sorted(variants)
        combinations = [ dict((varName, val) for varName, val in zip(varNames, prod))
                         for prod in it.product(*(variants[varName] for varName in varNames))]
        best_valid_f1 = 0.0
        combination_counter = 0
        for combination in combinations:
            clf = svm.SVC(gamma=combination['gamma'], C=combination['C'], kernel=combination['kernel'], probability=True)
            X_train = np.asarray([data[0] for data in svm_train_data])
            y_train = np.asarray([data[1] for data in svm_train_data])
            clf.fit(X_train, y_train)
            X_valid = np.asarray([data[0] for data in svm_dev_data])
            y_valid = np.asarray([data[1] for data in svm_dev_data])
            y_pred = clf.predict(X_valid)
            valid_f1 = getMacroFScore(y_pred, y_valid)
            valid_f1_score = getMacroFScore(y_pred, y_valid)[-1]
            if valid_f1_score > best_valid_f1:
                best_valid_f1 = valid_f1_score
                best_clf = clf
                best_combination_counter = combination_counter
            combination_counter+=1
    
        result_dir = os.path.join(results_path, "results_" + str(svm_type))
        ensure_directory(result_dir)
    
        # save the best model
        joblib.dump(best_clf, os.path.join(result_dir, "best_clf_" + experiment_name + str(svm_type)))
        X_valid = np.asarray([data[0] for data in svm_dev_data])
        y_valid = np.asarray([data[1] for data in svm_dev_data])
        y_pred = best_clf.predict(X_valid)
        y_predict_proba = best_clf.predict_proba(X_valid)
        valid_f1_score = getMacroFScore(y_pred, y_valid)
        best_combination = combinations[best_combination_counter]

        y_results = []
        for i in range(len(y_valid)):
            y_results.append([y_valid[i],y_pred[i], max(y_predict_proba[i])])
        load_save_pkl.save_as_pkl(y_valid, os.path.join(result_dir, "y_valid" + str(svm_type)))
        load_save_pkl.save_as_pkl(y_pred, os.path.join(result_dir, "y_pred" + str(svm_type)))
        load_save_pkl.save_as_pkl(y_results, os.path.join(result_dir, "y_results"+ str(svm_type)))
    
        with open(os.path.join(result_dir, "results_" + str(svm_type) + ".txt"),'a') as results_file:
            results_file.write("svm_type: " + str(svm_type) + "\n")
            results_file.write("best hyper parameters: " + "kernel:" + best_combination['kernel'] + ", " + "C:" + str(best_combination['C']) + ", " + "gamma:" + str(best_combination['gamma']) + "\n")
            results_file.write("Validation f1 score: " + str(best_valid_f1) + "\n")
            results_file.write("Test f1 score: " + str(valid_f1_score) + "\n")
    
    
        y_results = load_save_pkl.load_pickle_file(os.path.join(result_dir, "y_results" + str(svm_type)))
    
        with open(os.path.join(result_dir, "results_stats" + str(svm_type) + ".txt"),'a') as results_file1:
            header = "true" + ',' + "prediction" + ',' + "probability"
            results_file1.write(header)
            results_file1.write("\n")
            for item in y_results:
                results_file1.write(','.join(map(str, item)))
                results_file1.write("\n")
    
        # Prediction on test dataset
        X_test = np.asarray([data[0] for data in svm_test_data])
        y_test = np.asarray([data[1] for data in svm_test_data])

        y_pred_test = best_clf.predict(X_test)
        y_predict_proba_test = best_clf.predict_proba(X_test)
        test_f1_score = getMacroFScore(y_pred_test, y_test)
        print("Test f1 score %s" %test_f1_score)
    
        y_results_test = []
        for i in range(len(y_test)):
            y_results_test.append([y_test[i],y_pred_test[i], max(y_predict_proba_test[i])])
        load_save_pkl.save_as_pkl(y_test, os.path.join(result_dir, "y_test" + str(svm_type)))
        load_save_pkl.save_as_pkl(y_pred_test, os.path.join(result_dir, "y_pred_test" + str(svm_type)))
        load_save_pkl.save_as_pkl(y_results_test, os.path.join(result_dir, "y_results_test"+ str(svm_type)))
    
        with open(os.path.join(result_dir, "results_" + str(svm_type) + ".txt"),'a') as results_file:
            results_file.write("Test f1 score: " + str(test_f1_score) + "\n")
    
        y_results_test = load_save_pkl.load_pickle_file(os.path.join(result_dir, "y_results_test" + str(svm_type)))
    
        with open(os.path.join(result_dir, "results_stats_test" + str(svm_type) + ".txt"),'a') as results_file2:
            header = "true" + ',' + "prediction" + ',' + "probability"
            results_file2.write(header)
            results_file2.write("\n")
            for item in y_results_test:
                results_file2.write(','.join(map(str, item)))
                results_file2.write("\n")