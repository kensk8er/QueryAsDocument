"""
Executable file for exporting job list into csv file.

When running this file, 'working directory' need to be specified as Project Root (JobPostRecommendation).
"""
import json
import csv

__author__ = 'kensk8er'

if __name__ == '__main__':
    FILE = open('data/job/job_data.json', 'r')
    job_list = json.load(FILE)
    FILE.close()

    writer = csv.writer(file('data/job_list.csv', 'w'))
    writer.writerow(
        ['id', 'job name', 'company_name', 'job url', 'company url', 'job description', 'company description'])
    id = 0
    for job in job_list.values():
        writer.writerow(
            [id, job['job_name'].encode('utf-8'), job['company_name'].encode('utf-8'), job['job_url'].encode('utf-8'),
             job['company_url'].encode('utf-8'), job['job_description'].encode('utf-8'),
             job['company_description'].encode('utf-8')])
        id += 1
