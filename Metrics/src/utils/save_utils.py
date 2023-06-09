import os
import csv
import sys
from datetime import datetime


def save_results(config, metrics, name=None):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    if not os.path.exists(config['save_results_path']):
        os.mkdir(config['save_results_path'])

    save_file_name = config['exp_type'] + "_" + config['save_file_name'] + "_" + config['dataset_name'] + "_" + config[
        'model_name'] + '.csv'

    if not os.path.exists(os.path.join(config['save_results_path'], save_file_name)):
        f = open(os.path.join(config['save_results_path'], save_file_name), "a")
        f.close()
    result_file_path = os.path.join(config['save_results_path'], save_file_name)

    with open(result_file_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Experiment Name:', config['exp_type']])
        writer.writerow(['Dataset Name:', config['dataset_name']])
        writer.writerow(['Model Name', config['model_name']])
        writer.writerow(['Sub-Folder', name])
        writer.writerow(['Record Time:', current_time])
        writer.writerow([])
        for key, value in metrics.items():
            writer.writerow([key, value.data.item()])
        writer.writerow([])
        writer.writerow([])
    print("\n"
          "Saved metric in {}."
          "\n".format(config['save_results_path']))

    # with open(result_file_path) as csv_file:
    #     reader = csv.reader(csv_file)
    #     mydict = dict(reader)
