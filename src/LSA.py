"""
Executable file for LSA (Latent Semantic Analysis).

When running this file, 'working directory' need to be specified as Project Root (JobPostRecommendation).
"""
from glob import glob
import json
import os
import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from utils.util import enpickle

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


if __name__ == '__main__':
    # parameters
    n_components = 150  # this value need be less than the number of job posts
    n_results = 10  # this value need be less than the number of job posts

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
    lsa = TruncatedSVD(n_components=n_components)
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
    print 'pick up', n_results, 'most similar job posts for each resume...'
    results = get_n_most_similar_job_posts(similarity_matrix=similarities,
                                           n=n_results,
                                           resume_index_list=range(n_resume))  # resumes come after job posts

    print 'show recommendation results for each resume:\n'
    show_recommendation_results(result_lists=results, resume_indices=resume_indices, job_indices=job_indices)

    # calculate each metric based on relevancy judgements
    print 'load relevancy judgements...'
    relevancy_judgements =
