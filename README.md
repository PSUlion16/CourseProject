# CTOOMBS CS410 Course Project Documentation

#Description of this repository:

	answer.txt
		- Classified answers based off the test.jsonl document provided to team
	classify.py
		- final code for the Doc2Vec Twitter Sarcasm model
	CS410_Classification_Demo LINK!!!
		- 20 Minute Demonstration of project, code, issues
			- https://mediaspace.illinois.edu/media/t/1_fltka8gr
	project_proposal_ctoombs2.pdf
		- proposal for project
	project_status_report_chris_toombs.pdf
		- Status report from November in regards to project progress
	README.md
		- Full description and documentation for code
	TwitterSarcasmModel.d2v
		- Doc2Vec model from successful submission of project

#To Run the classifier:

	FROM PYCHARM:
		- You can load in the Project Repo AS-IS and run the code directly without modification
			- Most environments will already have NLTK and GENSIM included. If you get an error, just import those libraries to your environment
	FROM CLI:
		- Navigate to local folder for course project repo
		- type classifier.py
			NOTE: THIS ASSUMES YOU HAVE PYTHON INSTALLED ON YOUR COMPUTER

# Description of Code

	My classify utilizes both GENSIM and NLTK to classify the test data into SARCASTIC or NON_SARCASTIC tweets. I was able
	to use the TwitterTokenizer to parse the REPONSE fields of the data into Tokenized lists, with Handles removed, words moved to lowercase,
	and repeated characters shortened to one. 

	A brief description of the code methods:

		read_inputs() -- Reads in both TRAIN and TEST Files and tokenizes into lists
    		initialize_doc2vec() -- Initializes the DOC2VEC Model based off the TaggedDocument objects in the TRAIN list
    		get_labels() -- Determines the labels for the TEST dataset based off the DOC2VEC model saved in initialize_doc2vec()
					This method uses the INFER_VECTOR method of doc2vec to vectorize the test tweets
    		output_results() -- Outputs the data to answer.txt

# Difficulties
	
	The most difficult part of this project was the hyperparameter tuning. Much of the parameters had vague documentation and it seemed like
	use cases for parameter values varied greatlywith different users online. I probably spent 10-12 hours or so toying around with the parameters
	adding / removing items and observing behavior before landing on the set that worked for me. In the future, it may be more useful to find a more
	recent model with a larger user base, as it seems doc2vec is not as widely used (the git repo has not had a commit since 2018). I was confortable with
	the implementation of this as I have used word2vec in previous projects.

# References

	I saw a ton of forums for where people were discussing use cases, but the main sources I pulled from were the following:

	stackoverflow.com
	https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/doc2vec.py
	https://www.nltk.org/
	https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
