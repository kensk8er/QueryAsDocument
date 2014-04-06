"""
Executable file for LSA (Latent Semantic Analysis).

When running this file, 'working directory' need to be specified as Project Root (JobPostRecommendation).
"""
from glob import glob
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np

__author__ = 'kensk8er'


def read_text(directory_path):
    """
    Read all the text located under the specified directory.

    :param directory_path: The path for a directory that those files you'd like to read are located.
    :return: list[str]
    """
    text_list = []
    files = glob(directory_path + '/' + '*.txt')

    for file in files:
        text = open(file, 'r').read()
        text_list.append(text)

    return text_list


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


def get_n_most_similar_job_posts(similarity_matrix, resume_indices, n=10):
    """
    Return indices of n-most similar job posts and their cosine similarities.

    :param similarity_matrix:
    :param n:
    :param resume_indices: indices that specify which rows correspond to resume
    :return: list[list[tuple(job_post_index, cosine_similarity)]]
    """
    return_list = []

    for resume_index in resume_indices:
        similarities = zip(similarity_matrix[resume_index, :].tolist()[0], range(similarity_matrix.shape[0]))
        similarities.sort()
        similarities.reverse()

        n_result = 0
        result = []
        #while n_result < n:
        for similarity in similarities:
            job_post_index = similarity[1]

            if not job_post_index in resume_indices:  # only job posts can be valid results
                cosine_similarity = similarity[0]
                result.append((job_post_index, cosine_similarity))
                n_result += 1

                if n_result >= n:
                    break

        return_list.append(result)

    return return_list


def show_results(result_lists, n_resume):
    # note that this 'resume_indices' is different from the one in 'get_n_most_similar_job_posts'
    resume_indices = range(n_resume)
    for (result, resume_index) in zip(result_lists, resume_indices):
        print '[ resume:', resume_index, ']'
        rank = 1
        for row in result:
            job_post = row[0]
            similarity = row[1]
            print 'rank %s: %s (%s)' % (rank, job_post, similarity)
            rank += 1
        print ''


if __name__ == '__main__':
    # parameters
    n_components = 5  # this value need be less than the number of job posts
    n_results = 5  # this value need be less than the number of job posts

    # load job post data
    print 'read job post data...'
    job_text = read_text('data/toy/job')
    n_job = len(job_text)

    # load resume data
    print 'read resume data...'
    resume_text = read_text('data/toy/resume')
    n_resume = len(resume_text)
    n_text = n_job + n_resume

    # combine job post and resume data
    text = resume_text + job_text

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
    similarities = calculate_similarities(X)

    # pick up n-most similar job posts and show them
    print 'pick up', n_results, 'most similar job posts for each resume...'
    results = get_n_most_similar_job_posts(similarity_matrix=similarities,
                                           n=n_results,
                                           resume_indices=range(n_job, n_text))  # resumes comes after job posts

    print 'show results for each resume:\n'
    show_results(result_lists=results, n_resume=n_resume)
