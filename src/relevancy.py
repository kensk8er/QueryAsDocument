"""
Import relevancy judgements from csv file and save them into pickle file.

* Those pickle files are later used in LSA.py
When running this file, 'working directory' need to be specified as Project Root (JobPostRecommendation).
"""
import csv
from glob import glob
import json
import os
from utils.util import enpickle

__author__ = 'kensk8er'


def read_files(directory_path):
    """
    Read all the text located under the specified directory.

    :param directory_path: The path for a directory that those files you'd like to read are located.
    :return: dict[dict{name, text}]
    """
    return_dict = {}
    files = glob(directory_path + '/' + '*.csv')

    for FILE in files:
        dir_name, resume_name = os.path.split(FILE)
        resume_name = resume_name.split('.')[0]
        TEXT = open(FILE, 'r').read()
        return_dict[resume_name] = {}
        return_dict[resume_name]['name'] = resume_name
        return_dict[resume_name]['text'] = TEXT

    return return_dict


if __name__ == '__main__':
    relevancy_dict = {}

    FILE = open('data/job/job_data.json', 'r')
    job_data = json.load(FILE)
    assert isinstance(job_data, dict), 'job_data must be dict'
    FILE.close()

    file_names = glob('data/relevancy/*.csv')
    for file_name in file_names:
        FILE = open(file_name, 'rb')
        reader = csv.reader(FILE)
        row_num = 0
        resume_name = ''

        for row in reader:
            if row_num == 0:
                resume_name = row[0]

            if row_num == 1:
                pass

            if row_num >= 2:
                relevancy = row[0]
                job_url = row[1]

                # search job_name by job_url
                job_name = ''
                for key1, value1 in job_data.items():
                    if value1['job_url'] == job_url:
                        job_name = key1

                relevancy_dict[(resume_name, job_name)] = relevancy

            row_num += 1

    enpickle(relevancy_dict, 'data/relevancy/relevancy.pkl')
