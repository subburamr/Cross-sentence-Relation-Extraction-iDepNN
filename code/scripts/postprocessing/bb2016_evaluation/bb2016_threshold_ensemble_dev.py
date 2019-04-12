from __future__ import print_function
import os
import re
import zipfile
import load_save_pkl
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import glob
from collections import defaultdict, Counter
import shutil

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def get_predictions_prob_dev(results_dir):
    dev_predictions_file = results_dir + '*.decision_dev'
    dev_predictions_file = glob.glob(dev_predictions_file)[0]
    devfile_predictions_data = open(dev_predictions_file).readlines()
    devfile_predictions_data = [int(elem.strip())for elem in devfile_predictions_data]

    dev_predictions_prob_file = results_dir + '*.prob_dev'
    dev_predictions_prob_file = glob.glob(dev_predictions_prob_file)[0]
    devfile_predictions_prob_data = open(dev_predictions_prob_file).readlines()
    devfile_predictions_prob_data = [float(elem.strip())for elem in devfile_predictions_prob_data]
    return devfile_predictions_data, devfile_predictions_prob_data

def get_svm_predictions_prob_dev(results_dir):
    svm_predictions_file = results_dir + 'results_stats3.txt'
    svm_results_file_list = open(svm_predictions_file, 'rb').readlines()
    svm_devfile_predictions_data = [int(elem.split(',')[1]) for i, elem in enumerate(svm_results_file_list) if i >0]
    svm_devfile_predictions_prob_data = [float(elem.split(',')[2].strip()) for i, elem in enumerate(svm_results_file_list) if i >0]
    return svm_devfile_predictions_data, svm_devfile_predictions_prob_data

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_evaluation(train_k_type, dev_dataset_type, threshold):
    # labelled data
    svm_results_path = '../../data/svm_bionlp_2016/results/expt_3/' + train_k_type + '/test_' + dev_dataset_type + '/results_3/'
    birnn_results_path = '../../data/bionlp_2016/results/emnlp_dataset/brnn-fdp/postag_iob/' + train_k_type + '/test_' + dev_dataset_type + '/'
    idepnn_sdp_results_path = '../../data/bionlp_2016/results/emnlp_dataset/idepnn-sdp/postag_iob/' + train_k_type + '/test_' + dev_dataset_type + '/'
    idepnn_adp_results_path = '../../data/bionlp_2016/results/emnlp_dataset/idepnn-adp/postag_iob_with_h200/' + train_k_type + '/test_' + dev_dataset_type + '/'
    # print(birnn_results_path)
    # print(idepnn_sdp_results_path)
    # print(idepnn_adp_results_path)

    expt_input_path = 'D:/thesis_stuff/code/master_thesis_v2/emnlp_data_preparation/data/BB2016/BB2016_train_dev/' + dev_dataset_type + '/dev/'

    dev_final_data = load_save_pkl.load_pickle_file(os.path.join(expt_input_path, "BB2016_dev_data_w_others_" + dev_dataset_type + "_sdp_dep.pkl"))

    # annotations
    annotation_results_path = '../../data/bionlp_2016_full_ensemble_dev/annotations/emnlp_dataset/ensemble_with_threshold_' + str(threshold) + '/' + train_k_type + '/dev_' + dev_dataset_type + '/'

    ensure_directory(annotation_results_path)
    svm_devfile_predictions_data, svm_devfile_predictions_prob_data = get_svm_predictions_prob_dev(svm_results_path)
    birnn_devfile_predictions_data, birnn_devfile_predictions_prob_data = get_predictions_prob_dev(birnn_results_path)
    idepnn_sdp_devfile_predictions_data, idepnn_sdp_devfile_predictions_prob_data = get_predictions_prob_dev(idepnn_sdp_results_path)
    idepnn_adp_devfile_predictions_data, idepnn_adp_devfile_predictions_prob_data = get_predictions_prob_dev(idepnn_adp_results_path)
    print('Counters : svm %s \n birnn %s \n idepnn-sdp %s \n idepnn-adp %s' %(Counter(svm_devfile_predictions_data), Counter(birnn_devfile_predictions_data), Counter(idepnn_sdp_devfile_predictions_data), Counter(idepnn_adp_devfile_predictions_data)))

    key_file_dir = 'D:/thesis_stuff/datasets/BioNLPdata/BB3-data/BB3-event-rel/BioNLP-ST-2016_BB-event_dev/'
    count = 0
    predicted_rel_dict = defaultdict(list)
    for i, elem in enumerate(dev_final_data):
        elem=elem[0][0]
        if dev_dataset_type == 'k_0':
            instance_k_val = 0
        else:
            instance_k_val = 1
        if int(elem.split('::')[1]) <= instance_k_val:
            if threshold < 0.5:
                if (svm_devfile_predictions_data[i]==3 and 0.5 < svm_devfile_predictions_prob_data[i] <= 1-threshold) or (birnn_devfile_predictions_data[i]==1 and 0.5 < birnn_devfile_predictions_prob_data[i] <= 1-threshold) or                 (idepnn_sdp_devfile_predictions_data[i]==1 and 0.5 < idepnn_sdp_devfile_predictions_prob_data[i] <= 1-threshold) or                 (idepnn_adp_devfile_predictions_data[i]==1 and 0.5 < idepnn_adp_devfile_predictions_prob_data[i] <= 1-threshold):
                    count+=1
                    relation = 'Lives_In'
                    doc_id = elem.split('::')[0]
                    e1 = elem.split('::')[3]
                    e2 = elem.split('::')[4]
                    key_file = key_file_dir + doc_id
                    e1_token_type = elem.split('::')[6]
                    e2_token_type = elem.split('::')[7]
                    e1_token_num = elem.split('::')[8]
                    e2_token_num = elem.split('::')[9]
                    relation_id = 'R' + str((len(predicted_rel_dict[doc_id]) + 1))
                    predicted_rel_dict[doc_id].append([relation_id, relation, e1_token_type, e1_token_num, e2_token_type, e2_token_num])

            if (svm_devfile_predictions_data[i]==1 and svm_devfile_predictions_prob_data[i] >= threshold) or (birnn_devfile_predictions_data[i]==0 and birnn_devfile_predictions_prob_data[i] >= threshold) or (idepnn_sdp_devfile_predictions_data[i]==0 and idepnn_sdp_devfile_predictions_prob_data[i] >= threshold) or (idepnn_adp_devfile_predictions_data[i]==0 and idepnn_adp_devfile_predictions_prob_data[i] >= threshold):
                count+=1
                relation = 'Lives_In'
                doc_id = elem.split('::')[0]
                e1 = elem.split('::')[3]
                e2 = elem.split('::')[4]
                key_file = key_file_dir + doc_id
                e1_token_type = elem.split('::')[6]
                e2_token_type = elem.split('::')[7]
                e1_token_num = elem.split('::')[8]
                e2_token_num = elem.split('::')[9]
                relation_id = 'R' + str((len(predicted_rel_dict[doc_id]) + 1))
                predicted_rel_dict[doc_id].append([relation_id, relation, e1_token_type, e1_token_num, e2_token_type, e2_token_num])
    print('count %s' %count)
    counter = 0
    # write the result annotations
    for key, value in predicted_rel_dict.iteritems():
        with open(os.path.join(annotation_results_path, key + ".a2"),'wb') as ann_result_file:
            for rel_content in value:
                ann_result_file.write(rel_content[0] + "\t" + rel_content[1] + " " + "Bacteria" + ":" + rel_content[3] + " " + "Location" + ":" + rel_content[5])
                ann_result_file.write("\n")
                counter += 1

    print('num of extractions %s' %counter)

if __name__ == '__main__':
    # configurations
    threshold = 0.85 # [0.4, 0.5, 0.9, 0.95]

    train_k_type = 'train_k_le_1' # [train_k_0, train_k_le_1, train_k_le_2, train_k_le_3]
    dev_dataset_types = ['k_0', 'k_le_1']
    # dev_dataset_types = ['k_0', 'k_le_1', 'k_le_2', 'k_le_3', 'k_le_8', 'k_all']

    for dev_dataset_type in dev_dataset_types:
        run_evaluation(train_k_type, dev_dataset_type, threshold)

        annotation_results_path = '../../data/bionlp_2016_full_ensemble_dev/annotations/emnlp_dataset/ensemble_with_threshold_' + str(threshold) + '/' + train_k_type + '/dev_' + dev_dataset_type + '/'
        print('Creating zip ')
        print(annotation_results_path)
        shutil.make_archive(annotation_results_path, 'zip', annotation_results_path)