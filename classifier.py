# CTOOMBS 11/27/2019
#   This is the final project for CS410 Text Information systems
#   This program will read in tweets and classify them with a label of either SARCASM or NOT_SARCASM
#   The output of running this program will be a file called answer.txt
import logging
import os
import json
import random

# Tweet tokenizer for preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# Genism Libraries, used for vector space modeling
import gensim
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument

# Set Logging -- basic configuration
logging.basicConfig(format='%(asctime)s -- %(levelname)s: %(message)s',
                    level=logging.NOTSET,
                    datefmt='%Y-%m-%d %H:%M:%S')
my_logger = logging.getLogger('classifier_project')

# Global Variables for Classifier
test_docs = []  # List for the tweets, list of tagged document
train_docs = []  # Dictionary for training tweets, list of tagged document
class_labels = []
labels = {'SARCASM': 0, 'NOT_SARCASM': 1}
test_file = 'test.jsonl'
train_file = 'train.jsonl'


# This method will take in a filename and populate a list of dictionaries with tweets
def read_inputs() -> None:
    test_filepath = './data/' + test_file
    train_filepath = './data/' + train_file

    # Create a tweet tokenizer here. The one bad thing is we MAY need to preprocess new characters as in some
    # Tweets there is a different apostrophe
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    if os.path.exists(test_filepath) and os.path.exists(train_filepath):
        my_logger.info('Test File and Training File EXISTS!!!')
    else:
        my_logger.error('Input files do NOT EXIST! Input files are contained in the data directory')
        raise

    my_logger.info('Loading TEST file -- ' + test_file)

    with open(test_filepath, encoding='UTF-8') as file:  # Open File for Reading
        for input_line in file:  # For each line in the file, store off json as dict
            json_line = json.loads(input_line.strip('\n'))  # Place the line into a JSON object

            # Read the JSON response into a line to have stop words removed
            response = tokenizer.tokenize(json_line['response'])
            response_without_sw = [word for word in response]

            # Append results to the test docs set
            test_docs.append(response_without_sw)

    my_logger.info('TEST File Loaded. Total Tweets: ' + str(len(test_docs)))
    my_logger.info('Loading TRAINING file -- ' + train_file)

    with open(train_filepath, encoding='UTF-8') as file:  # Open File for Reading
        for input_line in file:  # For each line in the file, store off json as dict
            json_line = json.loads(input_line.strip('\n'))  # Place the line into a JSON object

            # Read the JSON response into a line to have stop words removed
            # and word not in string.punctuation
            response = tokenizer.tokenize(json_line['response'])
            response_without_sw = [word for word in response]

            # Create a new Tagged Document and place it at the end of the training docs
            label = labels.get(json_line['label'])

            # Create a new Tagged Document and place it at the end of the test docs, label for training is the class
            train_docs.append(TaggedDocument(words=response_without_sw, tags=[label]))

    my_logger.info('TRAINING File Loaded. Total Tweets: ' + str(len(train_docs)))

# This will initialize the doc2vec model which will be used for comparing the test data with the labeled train data
def initialize_doc2vec():
    my_logger.info('Initializing Doc to vec model ... ')
    # Create the training model, this is straight from Doc2Vec documentation, see for more options available
    # This will build the mode, build vocab, AND train the model
    # We can toy around with this to modify the fit, we shuffle our data one time to make sure to train intelligently
    random.shuffle(train_docs)
    model = doc2vec.Doc2Vec(documents=train_docs,
                                vector_size=100,
                                min_count=4,
                                alpha=0.0025,
                                epochs=100)

    # Save the model - you dont need to run the initialization each time once you've found a good model
    model.save('./TwitterSarcasmModel.d2v')

    my_logger.info('Doc2Vec Model Trained.')

# Infer the sarcasm sentiment using the docvecs MOST SIMILAR function. For training this seemed PRETTY close
def get_labels():
    model = doc2vec.Doc2Vec.load('./TwitterSarcasmModel.d2v')

    my_logger.info('Labeling Test Data ... ')

    # Replace this with train_docs if i want to run that
    for doc in test_docs: # train_docs
        inferred_vector = model.infer_vector(doc, epochs=250, alpha=0.020)  # doc.words

        # Sims if left printed as is, will give me the tag, as well as cosine similarity
        sims = model.docvecs.most_similar([inferred_vector], topn=1)

        # unpack numpy tuples as the sims are stored within pairs (label/similarity), we only need the label
        top_labels = [x for x,_ in sims]

        # Append the ending labels to the label list
        class_labels.append(top_labels[0])


# Print the results (twitter handle / label prediction) to answer.txt file
def output_results():
    with open('answer.txt', 'a') as file:  # We will always append to the file
        for idx, item in enumerate(class_labels):
            if item == 0:
                file.write('twitter_' + str(idx+1) + ',SARCASM' + "\n")
            else:
                file.write('twitter_' + str(idx+1) + ',NOT_SARCASM' + "\n")


# Main method - python will automatically run this
if __name__ == '__main__':
    read_inputs()
    initialize_doc2vec()
    get_labels()
    output_results()
