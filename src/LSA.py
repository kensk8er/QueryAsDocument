"""
Executable file for LSA (Latent Semantic Analysis).

When running this file, 'working directory' need to be specified as Project Root (JobPostRecommendation).
"""
import csv
from glob import glob
import json
import os
import datetime
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.pipeline import Pipeline
import numpy as np
import sys
from utils.util import enpickle, unpickle

__author__ = 'kensk8er'


def read_resume(directory_path):
    """
    Read all the text located under the specified directory.

    :param directory_path: The path for a directory that those files you'd like to read are located.
    :return: dict[dict{name, text}]
    """
    return_dict = {}
    files = glob(directory_path + '/' + '*.txt')

    for FILE in files:
        dir_name, resume_name = os.path.split(FILE)
        resume_name = resume_name.split('.')[0]
        TEXT = open(FILE, 'r').read()
        return_dict[resume_name] = {}
        return_dict[resume_name]['name'] = resume_name
        return_dict[resume_name]['text'] = TEXT

    return return_dict


def read_job_data(file_path):
    """
    Read the json file specified.

    :param file_path: The path for the json file you'd like to.
    :return: dict[dict{job_name, job_description, job_url, company_name, company_description, company_url}]
    """
    FILE = open(file_path, 'r')
    return_dict = json.load(FILE)
    FILE.close()

    return return_dict


def calculate_similarities(document_matrix):
    """
    Calculate the similarities between every document vector which is contained in the document matrix given as an
    argument.

    :rtype : matrix[float]
    :param document_matrix: Document Matrix whose each row contains a document vector.
    """
    # calculate inner products
    print 'calculate inner products...'
    inner_product_matrix = np.dot(document_matrix, document_matrix.T)

    # calculate norms
    print 'calculate norms...'
    norms = np.sqrt(np.multiply(document_matrix, document_matrix).sum(1))
    norm_matrix = np.dot(norms, norms.T)

    # calculate similarities
    print 'calculate similarities...'
    similarity_matrix = inner_product_matrix / norm_matrix

    return similarity_matrix


def get_n_most_similar_job_posts(similarity_matrix, resume_index_list, n=10):
    """
    Return indices of n-most similar job posts and their cosine similarities.

    :param similarity_matrix:
    :param n:
    :param resume_index_list: indices that specify which rows correspond to resume
    :return: list[list[tuple(job_post_index, cosine_similarity)]]
    """
    return_list = []
    n_resume = len(resume_index_list)

    for resume_index in resume_index_list:
        similarities = zip(similarity_matrix[resume_index, :].tolist()[0], range(similarity_matrix.shape[0]))
        similarities.sort()
        similarities.reverse()

        n_result = 0
        result = []
        #while n_result < n:
        for similarity in similarities:
            job_post_index = similarity[1]

            if not job_post_index in resume_index_list:  # only job posts can be valid results
                cosine_similarity = similarity[0]
                job_post_index -= n_resume  # convert to index only for job post
                result.append((job_post_index, cosine_similarity))
                n_result += 1

                if n_result >= n:
                    break

        return_list.append(result)

    return return_list


def show_recommendation_results(result_lists, resume_indices, job_indices):
    """
    Show the results in formatted way.

    :param result_lists: results to show
    :param n_resume: the number of resumes
    :param resume_indices:
    :param job_indices:
    """
    # note that this 'resume_indices' is different from the one in 'get_n_most_similar_job_posts'
    #resume_indices = range(n_resume)
    for (result, resume_name) in zip(result_lists, resume_indices):
        print '[ resume:', resume_name, ']'
        rank = 1
        for row in result:
            job_post_index = row[0]
            similarity = row[1]
            job_post_name = job_indices[job_post_index]['name']
            job_post_url = job_indices[job_post_index]['url']
            print 'rank %s: %s (similarity: %s, url: %s)' % (rank, job_post_name, similarity, job_post_url)
            rank += 1
        print ''


def convert_dict_list(dict_data):
    """
    Convert dictionary format data into list format data. Return both list format data and indices that show which list
    element corresponds to which dictionary element.

    :param dict_data: dict[dict]
    :return: list[str] list_data, list[str] indices
    """
    list_data = []
    indices = []

    for dict_datum in dict_data.values():
        if dict_datum.has_key('name'):
            # procedure for resume data
            list_data.append(unicode(dict_datum['text'], 'utf-8'))  # convert str into unicode
            indices.append(dict_datum['name'])
        else:
            # procedure for job data
            assert isinstance(dict_datum, dict)
            text_data = '\n'.join([dict_datum['job_name'], dict_datum['job_description'], dict_datum['company_name'],
                                   dict_datum['company_description']])
            list_data.append(text_data)
            indices.append({'name': dict_datum['job_name'], 'url': dict_datum['job_url']})

    return list_data, indices


def calculate_recall_precision_fscore(predicted, ground_truth, resume_names):
    scores = {}
    n_resume = len(resume_names)
    n_job = len(ground_truth[resume_names[0]])

    # iterate over every resume
    for resume_index in range(n_resume):
        resume_name = resume_names[resume_index]
        predicted_list = predicted[resume_index]
        predicted_labels = [0 for i in xrange(n_job)]  # 0 means irrelevant
        correct_labels = ground_truth[resume_name]

        # iterate over every recommended job post
        for job_post_id, similarity in predicted_list:
            predicted_labels[job_post_id] = 1  # 1 means relevant

        scores[resume_name] = precision_recall_fscore_support(correct_labels, predicted_labels, average='micro')

    return scores


def convert_relevancy_judgements(relevancy_judgements, job_indices, resume_names):
    n_job = len(job_indices)
    n_resume = len(resume_names)
    new_relevancy_judgements = {}

    # initialize new_relevancy_judgements
    for resume_name in resume_names:
        new_relevancy_judgements[resume_name] = [0 for i in xrange(n_job)]

    # convert job_indices into suitable format
    converted_job_indices = {}
    for job_post_id in xrange(n_job):
        converted_job_indices[job_indices[job_post_id]['name']] = job_post_id

    # iterate over every ground truth
    for key, relevancy in relevancy_judgements.items():
        resume_name = key[0]
        job_post_name = key[1]
        job_post_id = converted_job_indices[job_post_name]
        relevancy = int(relevancy)

        new_relevancy_judgements[resume_name][job_post_id] = relevancy

    return new_relevancy_judgements


def calculate_average_precision(predicted, ground_truth, resume_names):
    scores = {}
    n_resume = len(resume_names)
    n_result = len(predicted[0])

    # iterate over every resume
    for resume_index in range(n_resume):
        resume_name = resume_names[resume_index]
        predicted_list = predicted[resume_index]
        predicted_labels = [1 for i in range(n_result)]  # 1 means relevant
        correct_labels = []

        # iterate over every recommended job post
        for job_post_id, similarity in predicted_list:
            correct_labels.append(ground_truth[resume_name][job_post_id])

        scores[resume_name] = average_precision_score(correct_labels, predicted_labels)

    return scores


def calculate_mean_average_precision(average_precision_list):
    return np.mean(average_precision_list.values())


def calculate_mean_reciprocal_rank(predicted, ground_truth, resume_names):
    n_resume = len(resume_names)
    n_job_post = len(predicted[0])
    score = 0.

    # iterate over every resume
    for resume_index in range(n_resume):
        resume_name = resume_names[resume_index]
        predicted_list = predicted[resume_index]

        rank = 1
        # iterate over every result
        for job_post_id, similarity in predicted_list:
            if ground_truth[resume_name][job_post_id] == 1:  # 1 means relevant
                score += float(1) / rank
                break
            else:
                rank += 1

    score /= n_resume
    return score


def calculate_ndcg(predicted, ground_truth, resume_names):
    """
    Calculate the mean of Normalized Discounted Cumulative Gain.

    formula: DCG = rel_1 + Sigma_i_n(rel_i / log_2(i))
    """
    n_resume = len(resume_names)
    score = 0.
    # iterate over every resume
    for resume_index in range(n_resume):
        resume_name = resume_names[resume_index]
        predicted_list = predicted[resume_index]
        rank = 1
        dcg = 0.
        opt_dcg = 0.

        # iterate over every result
        for job_post_id, similarity in predicted_list:
            relevancy = ground_truth[resume_name][job_post_id]

            if rank == 1:
                dcg += relevancy
                opt_dcg += 1
            else:
                dcg += relevancy / math.log(rank, 2)
                opt_dcg += 1 / math.log(rank, 2)

            rank += 1

        score += dcg / opt_dcg

    score /= n_resume
    return score


if __name__ == '__main__':
    args = sys.argv

    # parameters
    n_components = range(10, 410, 10)  # this value need be less than the number of job posts
    n_result = 10  # this value need be less than the number of job posts

    os.system('rm result/result.csv')

    for n_component in n_components:
        print 'start procedure ( n_component =', n_component, ')'
        # load job post data
        print 'read job post data...'
        job_dict = read_job_data('data/job/job_data.json')
        n_job = len(job_dict)

        # load resume data
        print 'read resume data...'
        resume_dict = read_resume('data/resume')
        n_resume = len(resume_dict)
        n_text = n_job + n_resume

        # convert dictionary format into list format
        print 'convert dictionary into list format...'
        job_list, job_indices = convert_dict_list(job_dict)
        resume_list, resume_indices = convert_dict_list(resume_dict)

        # combine job post and resume data
        text = resume_list + job_list

        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = Pipeline((
            ('hasher', hasher),
            ('tf_idf', TfidfTransformer())  # TODO: you should try many different parameters here
        ))

        # calculate TF-IDF
        print 'calculate TF-IDF...'
        X = vectorizer.fit_transform(text)

        # perform LSA
        print 'perform LSA...'
        lsa = TruncatedSVD(n_components=n_component, algorithm='arpack')
        X = np.matrix(lsa.fit_transform(X))

        # calculate cosine similarities between each text
        print 'calculate cosine similarities...'
        similarities = calculate_similarities(X)

        print 'save similarities and indices...'
        date_time = datetime.datetime.today().strftime("%m%d%H%M%S")
        enpickle(similarities, 'cache/similarities_' + date_time + '.pkl')
        enpickle(resume_indices, 'cache/resume_indices_' + date_time + '.pkl')
        enpickle(job_indices, 'cache/job_indices_' + date_time + '.pkl')

        # pick up n-most similar job posts and show them
        print 'pick up', n_result, 'most similar job posts for each resume...'
        results = get_n_most_similar_job_posts(similarity_matrix=similarities,
                                               n=n_result,
                                               resume_index_list=range(n_resume))  # resumes come after job posts

        print 'show recommendation results for each resume:\n'
        show_recommendation_results(result_lists=results, resume_indices=resume_indices, job_indices=job_indices)

        # calculate each metric based on relevancy judgements
        print 'load relevancy judgements...'
        relevancy_judgements = unpickle('data/relevancy/relevancy.pkl')

        print 'convert relevancy judgements into appropriate format...'
        relevancy_judgements = convert_relevancy_judgements(relevancy_judgements, job_indices, resume_indices)

        # calculate recall, precision, and f-score
        # note that this precision is same as precision@k
        print 'calculate precision, recall, and fscore...'
        recall_precision_fscores = calculate_recall_precision_fscore(results, relevancy_judgements, resume_indices)
        enpickle(recall_precision_fscores, 'result/recall_precision_fscores.pkl')

        print 'calculate average precision...'
        average_precision = calculate_average_precision(results, relevancy_judgements, resume_indices)
        enpickle(average_precision, 'result/average_precision.pkl')

        print 'calculate mean-average prevision...'
        mean_average_precision = calculate_mean_average_precision(average_precision)
        enpickle(mean_average_precision, 'result/mean_average_precision.pkl')

        print 'calculate mean reciprocal rank...'
        mean_reciprocal_rank = calculate_mean_reciprocal_rank(results, relevancy_judgements, resume_indices)
        enpickle(mean_reciprocal_rank, 'result/mean_reciprocal_rank.pkl')

        print 'calculate mean NDCG...'
        mean_ndcg = calculate_ndcg(results, relevancy_judgements, resume_indices)
        enpickle(mean_ndcg, 'result/mean_ndcg.pkl')

        # output the results
        writer = csv.writer(file('result/result.csv', 'ab'))
        writer.writerow(['dimensions', 'precision@10 (kensk8er)', 'precision@10 (hansong)', 'precision@10 (yuchen)',
                         'mean average precision', 'mean reciprocal rank', 'mean NDCG@10'])
        writer.writerow([n_component, recall_precision_fscores['kensk8er'][0], recall_precision_fscores['hansong'][0],
                         recall_precision_fscores['yuchen'][0], mean_average_precision, mean_reciprocal_rank,
                         mean_ndcg])
        writer.writerow([''])
        print 'finish procedures'
        print ''
