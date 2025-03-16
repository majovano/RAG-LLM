import json
import datasets
import re
import nltk
import pandas as pd
import math
import numpy as np
import string
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def preprocess_function(text):
    # Convert text to lowercase
    text = text.lower()

    # Replace hyphens with spaces before removing other punctuation to preserve word breaks
    remove_hyphen_re = re.compile(r"-")
    text = remove_hyphen_re.sub(' ', text)

    # Remove punctuation
    remove_punctuation_re = re.compile(f"[{string.punctuation}]")
    text = re.sub(remove_punctuation_re,'', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    combined_stop_words = set(stopwords.words('english')) | set(ENGLISH_STOP_WORDS)
    # Filter out stop words and digits
    tokens = [token for token in tokens if token not in combined_stop_words and not token.isdigit()]

    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    processed_text = ' '.join(tokens)
    return processed_text

def combine_fields(example):
    
    #fields_to_combine = ['description']
    #fields_to_combine = ['name','description']
    #fields_to_combine = ['name','ingredients','description']
    fields_to_combine = ['name', 'steps', 'tags','ingredients','description']
    combined_text = ' '.join([example[field] for field in fields_to_combine])
    return {'combined': preprocess_function(combined_text)}

def query_knn_indices(query, vectorizer,nn):
    preprocessed_query = preprocess_function(query)
	# apply kneighbors to the query vector
    query_vector = vectorizer.transform([preprocessed_query])
    distances, indices = nn.kneighbors(query_vector)
    return distances, indices

## Retrieving the query
# Function to retrieve the ground truth for a given query
def get_ground_truth(query, queries_dict):
    for q in queries_dict["queries"]:
        if q["q"] == query:
            # return q["r"]
            return [r - 1 for r in q["r"]]
    return None

def get_all_queries(queries_dict):
	queries_list = []
	for q in queries_dict["queries"]:
		queries_list.append(q["q"])
	return queries_list

def evaluate_results(ground_truth, retrieved_results):
    try:
      # Calculate the number of relevant results
      num_relevant_results = len(set(ground_truth).intersection(set(retrieved_results)))
      # Calculate the precision
      precision = num_relevant_results / len(retrieved_results)
      # Calculate the recall
      recall = num_relevant_results / len(ground_truth)
      # Calculate the F1-score
      f1_score = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError: # when nothing is found
      precision = 0
      recall = 0
      f1_score = 0
    return precision, recall, f1_score


def macro_metrics(queries_dict,vectorizer,nn):

    macro_precision = 0
    macro_recall = 0
    macro_f1_score = 0
    queries = get_all_queries(queries_dict)
    for query in queries:
        # we get the ground truth here
        ground_truth = get_ground_truth(query,queries_dict)
        # now we get predictions
        distances, indices = query_knn_indices(query,vectorizer,nn)
        # convert into a list
        indices_list = indices[0].tolist()
        # now call for the metrics
        precision, recall, f1_score = evaluate_results(set(ground_truth), set(indices_list))
        macro_precision += precision
        macro_recall += recall
        macro_f1_score += f1_score
    return macro_precision/len(queries), macro_recall/len(queries), macro_f1_score/len(queries)

def micro_metrics(queries_dict,vectorizer,nn):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    queries = get_all_queries(queries_dict)

    for query in queries:
        # Get the ground truth for this query
        ground_truth = get_ground_truth(query,queries_dict)
        # Get predictions using a k-nearest neighbors query
        distances, indices = query_knn_indices(query,vectorizer,nn)
        # Convert indices to a list
        indices_list = indices[0].tolist()

        # Convert lists to sets for easier calculation
        ground_truth_set = set(ground_truth)
        predictions_set = set(indices_list)

        # Calculate true positives, false positives, and false negatives
        true_positives = len(ground_truth_set.intersection(predictions_set))
        false_positives = len(predictions_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predictions_set)

        # Update total counts
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

    # Calculate micro-averaged Precision
    if (total_true_positives + total_false_positives) > 0:
        micro_precision = total_true_positives / (total_true_positives + total_false_positives)
    else:
        micro_precision = 0

    # Calculate micro-averaged Recall
    if (total_true_positives + total_false_negatives) > 0:
        micro_recall = total_true_positives / (total_true_positives + total_false_negatives)
    else:
        micro_recall = 0

    # Calculate micro-averaged F1 Score
    if (micro_precision + micro_recall) > 0:
        micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    else:
        micro_f1_score = 0

    return micro_precision, micro_recall, micro_f1_score

def average_precision(ground_truth, retrieved_results):
	# Initialize variables to keep track of the number of relevant results and the total precision
    num_relevant_results = 0
    total_precision = 0
    if len(ground_truth) != 0:
        # Iterate over the retrieved results
        for i, result in enumerate(retrieved_results):
            # Check if the result is relevant
            if result in ground_truth:
                # Increment the number of relevant results
                num_relevant_results += 1
                # Calculate the precision at this rank
                precision = num_relevant_results / (i + 1)
    			# Add the precision to the total precision
                total_precision += precision
    	# Calculate the average precision
        avg_precision = total_precision / len(ground_truth)
        return avg_precision
    else:
        avg_precision = 0
    return avg_precision

# Calculate the mean average precision for all queries
def mean_average_precision(queries,queries_dict,vectorizer,nn):
	# Initialize a variable to keep track of the total average precision
	total_avg_precision = 0
	# Get all the queries
	# queries = get_all_queries()
	# Iterate over the queries
	for query in queries:
		# Get the ground truth for this query
		ground_truth = get_ground_truth(query,queries_dict)
		# Get predictions using a k-nearest neighbors query
		distances, indices = query_knn_indices(query,vectorizer,nn)
		# Convert indices to a list
		indices_list = indices[0].tolist()
		# Calculate the average precision for this query
		avg_precision = average_precision(ground_truth, indices_list)
		# Add the average precision to the total average precision
		total_avg_precision += avg_precision
	# Calculate the mean average precision
	mean_avg_precision = total_avg_precision / len(queries)
	return mean_avg_precision

def show_query_results(queries,vectorizer,nn,dataset):
    #
	retrieved_document = []
	for query in queries:
		# print(f"Query: {query}")
		distances, indices = query_knn_indices(query,vectorizer,nn)
		indices_list = indices[0].tolist()
		# only print the top k element for each query
		recipe = dataset[indices_list[0]]
		# print(f"Recipe Name: {recipe['name']}")
		# print(f"Ingredients: {recipe['ingredients']}")
		# print(f"Description: {recipe['description']}\n")
		# print("\n")
		retrieved_document.append(recipe)
	return retrieved_document


def lm():
  
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')

    dataset = datasets.load_dataset("parquet", data_files="./recipes.indexed.parquet")['train']
    queries_dict = json.load(open("./queries.json", "r"))
    queries = get_all_queries(queries_dict)
    # Load the language model
    nlp = spacy.load("en_core_web_sm") # For English

    digits = re.compile(r'\d')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Regular expressions for removing punctuation and hyphens
    remove_punctuation_re = re.compile(f"[{string.punctuation}]")
    remove_hyphen_re = re.compile(r"-")

    # Combine NLTK and sklearn stop words
    combined_stop_words = set(stopwords.words('english')) | set(ENGLISH_STOP_WORDS)
    
    dataset_with_combined = dataset.map(combine_fields)

    
    # Initialize TfidfVectorizer with the SpaCy-based preprocessor
    vectorizer = TfidfVectorizer(ngram_range= (1,3))

    # Fit and transform the descriptions
    n_neighbors=12
    tfidf_matrix = vectorizer.fit_transform(dataset_with_combined['combined'])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine',algorithm = 'brute')
    nn.fit(tfidf_matrix)

    query =  "What is a turducken?"
    distances, indices = query_knn_indices(query, vectorizer,nn)
    # indices_list = indices[0].tolist()
    # for idx in indices_list:

    #     recipe = dataset[idx]


        # print(f"Recipe Name: {recipe['name']}")
        # print(f"Ingredient: {recipe['steps']}")
        # print(f"Ingredient: {recipe['ingredients']}")
        # print(f"Description: {recipe['description']}\n")

    ground_truth = get_ground_truth(query, queries_dict)
    all_queries = get_all_queries(queries_dict)
    ground_truth = get_ground_truth(query, queries_dict)

    macro_precision, macro_recall, macro_f1_score = macro_metrics(queries_dict,vectorizer,nn)
    print(f"Macro Precision: {macro_precision}")

    print(f"Macro Recall: {macro_recall}")

    print(f"Macro F1-score: {macro_f1_score}")

    # Evaluate the results


    micro_precision, micro_recall, micro_f1_score = micro_metrics(queries_dict,vectorizer,nn)
    print(f"Micro Precision: {micro_precision}")

    print(f"Micro Recall: {micro_recall}")

    print(f"Micro F1-score: {micro_f1_score}")

    # Evaluate the results
    mean_avg_precision = mean_average_precision(queries,queries_dict,vectorizer,nn)
    print(f"Mean Average Precision: {mean_avg_precision}")

    # first 3 good retrieval other 3 not good retrieval
    sample_queries = ["How do I make za'atar?","Can I use pork to make shish kebab? If not, what alternative do you suggest?","What is a turducken?","Where can I follow cooking classes?","At what temperature should I preheat my oven to bake croquembouche?","What do I need for quesadillas?"]
    # the ten queries of exercise 8
    ten_queries = ["Retrieve the top-rated chocolate cake recipe","List all recipes that use zucchini but no egg",
               "Find low-calorie vegetarian breakfast recipes under 300 calories", "What are popular summer recipes using tomatoes?",
               "Show traditional Japanese desserts", "Find recipes that primarily use sous vide cooking",
               "How do I scale a recipe for lasagna to serve 20 people?", "What are the common criticisms found in reviews for recipes with tofu?",
               "Adapt a recipe for peanut butter cookies to be nut-free","Recommend a quick dinner recipe that takes less than 30 minutes to prepare"]
    # retrieved_document will be used in the prompt
    retrieved_document = show_query_results(sample_queries,vectorizer,nn,dataset)

    prompt = f"""

    YOUR PROMPT GOES HERE

    """
    irrelevant_context = """
    Richard Gary Brautigan (January 30, 1935 – c. September 16, 1984)
    was an American novelist, poet, and short story writer. A prolific writer,
    he wrote throughout his life and published ten novels, two collections of
    short stories, and four books of poetry. Brautigan's work has been published
    both in the United States and internationally throughout Europe, Japan,
    and China. He is best known for his novels Trout Fishing in America (1967),
    In Watermelon Sugar (1968), and The Abortion: An Historical Romance 1966 (1971).
    """
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto'
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # if you want add any of these in the input_string
    # input_string =  prompt + str(retrieved_ten_documents[9])   + irrelevant_context# + str(retrieved_document[0])
    input_string = prompt + sample_queries[0] 

    encoded_prompt = tokenizer(input_string, return_tensors="pt", add_special_tokens=False)
    encoded_prompt = encoded_prompt.to("cuda")


    generated_ids = model.generate(**encoded_prompt, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])



#### LSI BEGINS HERE
##### LSI BEGINS HERE

def preprocess_function_lsi(text):
    # Convert text to lowercase
    text = text.lower()

    # Replace hyphens with spaces before removing other punctuation to preserve word breaks
    text = remove_hyphen_re_lsi.sub(' ', text)

    # Remove punctuation
    text = re.sub(remove_punctuation_re_lsi,'', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Filter out stop words and digits
    tokens = [token for token in tokens if token not in combined_stop_words_lsi and not token.isdigit()]

    # Apply lemmatization
    lemmatized_tokens = [lemmatizer_lsi.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    processed_text = ' '.join(tokens)
    return processed_text

"""## Preprocessing before the vectorizer_lsi (Here combine the fields)"""

def combine_fields_lsi(example):
    # fields_to_combine = ['description']
    # fields_to_combine = ['name','description']
    # fields_to_combine = ['name', 'ingredients','description']
    fields_to_combine = ['name', 'steps', 'tags','ingredients','description']
    combined_text = ' '.join([example[field] for field in fields_to_combine])
    return {'combined': preprocess_function_lsi(combined_text)}


# Nearst Neighbour
from sklearn.neighbors import NearestNeighbors
def query_knn_indices_lsi(query):
    preprocessed_query = preprocess_function_lsi(query)

    # apply kneighbors to the query vector
    query_vector = vectorizer_lsi.transform([preprocessed_query])

    # Assuming `lsa` is your LSA pipeline (SVD + Normalizer)
    query_vector_lsi = lsi_lsi.transform(query_vector)

    distances, indices = nn_lsi.kneighbors(query_vector_lsi)
    return distances, indices

## Retrieving the query
# Function to retrieve the ground truth for a given query
def get_ground_truth_lsi(query):
    for q in queries_lsi["queries"]:
        if q["q"] == query:
            # return q["r"]
            return [r - 1 for r in q["r"]]
    return None

def get_all_queries_lsi():
	queries_list = []
	for q in queries_lsi["queries"]:
		queries_list.append(q["q"])
	return queries_list

# Now time to implement Recall, Precision and F1

def evaluate_results_lsi(ground_truth, retrieved_results):
    try:
      # Calculate the number of relevant results
      num_relevant_results = len(set(ground_truth).intersection(set(retrieved_results)))
      # Calculate the precision
      precision = num_relevant_results / len(retrieved_results)
      # Calculate the recall
      recall = num_relevant_results / len(ground_truth)
      # Calculate the F1-score
      f1_score = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError: # when nothing is found
      precision = 0
      recall = 0
      f1_score = 0
    return precision, recall, f1_score

def macro_metrics_lsi():

    macro_precision_lsi = 0
    macro_recall_lsi = 0
    macro_f1_score_lsi = 0
    queries_lsi = get_all_queries_lsi()
    for query in queries_lsi:
        # we get the ground truth here
        ground_truth = get_ground_truth_lsi(query)
        # now we get predictions
        distances, indices = query_knn_indices_lsi(query)
        # convert into a list
        indices_list = indices[0].tolist()
        # now call for the metrics
        precision, recall, f1_score = evaluate_results_lsi(set(ground_truth), set(indices_list))
        macro_precision_lsi += precision
        macro_recall_lsi += recall
        macro_f1_score_lsi += f1_score
    return macro_precision_lsi/len(queries_lsi), macro_recall_lsi/len(queries_lsi), macro_f1_score_lsi/len(queries_lsi)


def micro_metrics_lsi():
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    queries_lsi = get_all_queries_lsi()

    for query in queries_lsi:
        # Get the ground truth for this query
        ground_truth = get_ground_truth_lsi(query)
        # Get predictions using a k-nearest neighbors query
        distances, indices = query_knn_indices_lsi(query)
        # Convert indices to a list
        indices_list = indices[0].tolist()

        # Convert lists to sets for easier calculation
        ground_truth_set = set(ground_truth)
        predictions_set = set(indices_list)

        # Calculate true positives, false positives, and false negatives
        true_positives = len(ground_truth_set.intersection(predictions_set))
        false_positives = len(predictions_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predictions_set)

        # Update total counts
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

    # Calculate micro-averaged Precision
    if (total_true_positives + total_false_positives) > 0:
        micro_precision = total_true_positives / (total_true_positives + total_false_positives)
    else:
        micro_precision = 0

    # Calculate micro-averaged Recall
    if (total_true_positives + total_false_negatives) > 0:
        micro_recall = total_true_positives / (total_true_positives + total_false_negatives)
    else:
        micro_recall = 0

    # Calculate micro-averaged F1 Score
    if (micro_precision + micro_recall) > 0:
        micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    else:
        micro_f1_score = 0

    return micro_precision, micro_recall, micro_f1_score

# MAP Metrics for LSI
# Calculate the average precision for a single query
def average_precision_lsi(ground_truth, retrieved_results):
    # Initialize variables to keep track of the number of relevant results and the total precision
    num_relevant_results = 0
    total_precision = 0
    if len(ground_truth) != 0:
        # Iterate over the retrieved results
        for i, result in enumerate(retrieved_results):
            # Check if the result is relevant
            if result in ground_truth:
                # Increment the number of relevant results
                num_relevant_results += 1
                # Calculate the precision at this rank
                precision = num_relevant_results / (i + 1)
                # Add the precision to the total precision
                total_precision += precision
        # Calculate the average precision
        avg_precision = total_precision / len(ground_truth)
        return avg_precision
    else:
        avg_precision = 0
    return avg_precision

# Calculate the mean average precision for all queries
def mean_average_precision_lsi():
	# Initialize a variable to keep track of the total average precision
	total_avg_precision = 0
	# Get all the queries
	queries_lsi = get_all_queries_lsi()
	# Iterate over the queries
	for query in queries_lsi:
		# Get the ground truth for this query
		ground_truth = get_ground_truth_lsi(query)
		# Get predictions using a k-nearest neighbors query
		distances, indices = query_knn_indices_lsi(query)
		# Convert indices to a list
		indices_list = indices[0].tolist()
		# Calculate the average precision for this query
		avg_precision = average_precision_lsi(ground_truth, indices_list)
		# Add the average precision to the total average precision
		total_avg_precision += avg_precision
	# Calculate the mean average precision
	mean_avg_precision = total_avg_precision / len(queries_lsi)
	return mean_avg_precision

def lsi():
  global remove_hyphen_re_lsi
  global remove_punctuation_re_lsi
  global combined_stop_words_lsi
  global lemmatizer_lsi
  global queries_lsi
  global vectorizer_lsi
  global lsi_lsi
  global nn_lsi
  dataset = datasets.load_dataset("parquet", data_files="./recipes.indexed.parquet")['train']
  queries_lsi = json.load(open("./queries.json", "r"))

  nltk.download('wordnet')
  nltk.download('stopwords')
  nltk.download('punkt')


  # Load the language model
  nlp = spacy.load("en_core_web_sm") # For English


  digits = re.compile(r'\d')

  lemmatizer_lsi = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))


  # Initialize the lemmatizer_lsi
  lemmatizer_lsi = WordNetLemmatizer()

  # Regular expressions for removing punctuation and hyphens
  remove_punctuation_re_lsi = re.compile(f"[{string.punctuation}]")
  remove_hyphen_re_lsi = re.compile(r"-")

  # Combine NLTK and sklearn stop words
  combined_stop_words_lsi = set(stopwords.words('english')) | set(ENGLISH_STOP_WORDS)
  
  
  dataset_with_combined = dataset.map(combine_fields_lsi)



  """## TF-IDF"""
  # Initialize TfidfVectorizer with the SpaCy-based preprocessor
  vectorizer_lsi = TfidfVectorizer()

  # Fit and transform the descriptions
  # combined_df = pd.concat([dataset, ['name', 'ingredients', 'steps', 'tags', 'description']], axis=1)
  tfidf_matrix = vectorizer_lsi.fit_transform(dataset_with_combined['combined'])

  # Define the number of components
  n_components = 1000
  
  # initialize and fit the truncatedSVD
  svd = TruncatedSVD(n_components=n_components)
  # fit the SVD
  lsi_lsi = make_pipeline(svd, Normalizer(copy=False))
  # fit the LSA
  lsi_matrix = lsi_lsi.fit_transform(tfidf_matrix)

  # compute cosine similarities

  query = "What kind of roux do I need for gumbo?"

  preprocessed_query = preprocess_function_lsi(query)

  # apply kneighbors to the query vector
  query_vector = vectorizer_lsi.transform([preprocessed_query])

  # Assuming `lsa` is your LSA pipeline (SVD + Normalizer)
  query_vector_lsi = lsi_lsi.transform(query_vector)

  # cosine_similarities[0]

  # Nearst Neighbour
  n_neighbors = 12
  nn_lsi = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
  nn_lsi.fit(lsi_matrix)

  ground_truth = get_ground_truth_lsi(query)
  all_queries = get_all_queries_lsi()




  # Evaluate the results
  macro_precision_lsi, macro_recall_lsi, macro_f1_score_lsi = macro_metrics_lsi()
  print(f"Macro_Precision_LSI: {macro_precision_lsi}")

  print(f"Macro_Recall_LSI: {macro_recall_lsi}")

  print(f"Macro_F1-score_LSI: {macro_f1_score_lsi}")

  # Micro Metrics for LSI
  # Evaluate the results


  micro_precision, micro_recall, micro_f1_score = micro_metrics_lsi()
  print(f"Micro_Precision_LSI: {micro_precision}")

  print(f"Micro_Recall_LSI: {micro_recall}")

  print(f"Micro_F1-score_LSI: {micro_f1_score}")


  # Evaluate the results
  lsi_mean_avg_precision = mean_average_precision_lsi()
  print(f"LSI Mean Average Precision: {lsi_mean_avg_precision}")

  # !pip freeze > requirements-lsi.txt

if __name__ == "__main__":

    print("Starting LSI")
    lsi()
    print("Finishing LSI")
    print("Starting LM")
    lm()
    print("Finishing LM")



"""

#######Q1


import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import datasets


# R_NUMBER_SEED = 1234567 # Replace this with your own student number
R_NUMBER_SEED = 928036 # my student number
DOCS_TO_ADD = 1000
query_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_queries.parquet")["train"]
all_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_full.parquet")["train"]
# Shuffle with seed and take only n docs
shuffled_documents = all_documents.shuffle(seed=R_NUMBER_SEED)
random_documents = shuffled_documents.select(range(DOCS_TO_ADD))
# Concatenate relevant documents with random sample and shuffle again
anthology_sample = datasets.concatenate_datasets([query_documents, random_documents]).shuffle(seed=R_NUMBER_SEED)
# Export to Parquet to avoid downloading full anthology
anthology_sample.to_parquet("./anthology_sample.parquet")


# Remove stopwords
def remove_stopwords(doc):
    text = ' '.join([word for word in doc.split() if word.lower() not in stop_words])

def preprocess_document(doc):
    # Flatten dictionary and combine relevant text fields
    text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('full_text', '')}"
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    
    return text

def preprocess_documents(documents):
    return [preprocess_document(doc) for doc in documents]

preprocessed_documents = preprocess_documents(anthology_sample)


import spacy
from spacy.pipeline import Sentencizer
import time
# Load the SpaCy model
# nlp = spacy.load("en_core_web_sm") ### SLOW


# nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
# nlp.enable_pipe("senter")
# nlp.enable_pipe("parser")
# for doc in nlp.pipe(texts, n_process=4):
# Load SpaCy model
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
# Add the sentencizer to the pipeline
if not nlp.has_pipe("sentencizer"):
    nlp.add_pipe("sentencizer")

start_time = time.time()
def preprocess_documents(documents):
    all_sentences = []
    all_sentences_to_doc_map = []
    count = 1
    for doc in documents:
        if count % 10 == 0:
            print("doc:",count)        # print(doc["acl_id"])
        full_text = f"{doc.get('full_text')}"
        spacy_doc = nlp(full_text)
        for sent in spacy_doc.sents:
            all_sentences.append(sent.text)
            all_sentences_to_doc_map.append(doc["acl_id"])
        count += 1
    return all_sentences,all_sentences_to_doc_map

all_sentences, all_sentences_to_doc_map = preprocess_documents(anthology_sample)
spacy_time = time.time() - start_time
print(f"Spacy extract time: {spacy_time} seconds")



# Initialize lists to store embeddings
# Initialize the MiniLM model for document embeddings
minilm_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Function to get document embedding using MiniLM
def get_document_embedding(document):
    return minilm_model.encode(document, convert_to_tensor=False)

minilm_embeddings = []
# bert_embeddings = []


# preprocessed_documents = preprocess_documents(anthology_sample)
# minilm_embeddings2 = [get_mpnet_embedding(doc) for doc in preprocessed_documents[:10]]  # Only first 1000 for k-NN
from sklearn.neighbors import NearestNeighbors
# knn2 = NearestNeighbors(n_neighbors=5, metric='cosine').fit(minilm_embeddings)

# Fit NearestNeighbors model
nn_minilm = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
nn_minilm.fit(minilm_embeddings)



import numpy as np

## Retrieving the query
# Function to retrieve the ground truth for a given query
def get_ground_truth(query):
    for q in queries["queries"]:
        if q["q"] == query:
            # return q["r"]
            return [r for r in q["r"]]
    return None
# ground_truth = get_ground_truth(query)

# Function to compare the result with the ground truth for a single query
# Function to calculate average precision for a single query
def average_precision(retrieved_docs, ground_truth_ids):
    if not ground_truth_ids:
        return 0
    retrieved_docs_set = set(retrieved_docs)
    ground_truth_set = set(ground_truth_ids)
    
    num_relevant = 0
    precision_sum = 0
    for i, doc in enumerate(retrieved_docs):
        if doc in ground_truth_set:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    
    return precision_sum / len(ground_truth_set)

# Function to compare the result with the ground truth for a single query
def compare_with_ground_truth(query_text, nn_model, dataset, ground_truth_function, k):
    # print("cgt k=",k)
    # Get nearest neighbors
    indices = get_nearest_neighbors(query_text, nn_model, k)
    
    # Convert numpy.int64 to Python int
    indices = [int(i) for i in indices]
    
    # Retrieve document IDs for nearest neighbors
    retrieved_docs = [dataset[i]['acl_id'] for i in indices]
    
    # Get ground truth
    ground_truth_ids = ground_truth_function(query_text)
    
    if not ground_truth_ids:
        return 0, 0, 0, 0, 0, 0, 0  # Return zeros if no ground truth is available
    
    # Calculate true positives, false positives, and false negatives
    tp = len(set(retrieved_docs) & set(ground_truth_ids))
    fp = len(retrieved_docs) - tp
    fn = len(ground_truth_ids) - tp
    
    # Calculate precision, recall, and F1-score
    precision = tp / len(retrieved_docs) if retrieved_docs else 0
    recall = tp / len(ground_truth_ids) if ground_truth_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    # Calculate average precision
    ap = average_precision(retrieved_docs, ground_truth_ids)
    
    return precision, recall, f1, ap, tp, fp, fn

# Function to evaluate the model on all queries
def evaluate_model_on_all_queries(queries, embeddings, dataset, ground_truth_function, k):
    # Fit NearestNeighbors model
    print("k=",k)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(embeddings)

    precision_scores = []
    recall_scores = []
    f1_scores = []
    ap_scores = []
    total_tp = total_fp = total_fn = 0

    for query in queries["queries"]:
        query_text = query["q"]
        precision, recall, f1, ap, tp, fp, fn = compare_with_ground_truth(query_text, nn, dataset, ground_truth_function, k)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ap_scores.append(ap)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate macro average metrics
    macro_precision = np.mean(precision_scores)
    macro_recall = np.mean(recall_scores)
    macro_f1 = np.mean(f1_scores)
    
    # Calculate micro average metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0
    
    # Calculate mean AP
    mean_ap = np.mean(ap_scores)
    
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap


def get_nearest_neighbors(query_text, nn_model, k):
    # query_embedding = get_aggregated_word_embeddings([query_text])
    # print(query_text)
    # test_query_word_embed = get_aggregated_word_embeddings(query_text).reshape(1, -1)
    test_query_word_embed = get_document_embedding(query_text).reshape(1, -1)
    distances, indices = nn_model.kneighbors(test_query_word_embed, n_neighbors=k)
    return indices[0]

def nearest_neighbour(embeddings): # can be tfidf_matrix or LSI_matrix doesnt
    nn = NearestNeighbors(n_neighbors=1, metric='cosine',algorithm = 'brute')
    nn.fit(embeddings)
    
# test_query = queries["queries"][0]['q']
# get_nearest_neighbors(test_query,nn_minilm,1)

k = 5
macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap = evaluate_model_on_all_queries(queries, minilm_embeddings, anthology_sample, get_ground_truth, k)
print(f"Macro Average Precision: {macro_precision:.4f}")
print(f"Macro Average Recall: {macro_recall:.4f}")
print(f"Macro Average F1-Score: {macro_f1:.4f}")
print(f"Micro Average Precision: {micro_precision:.4f}")
print(f"Micro Average Recall: {micro_recall:.4f}")
print(f"Micro Average F1-Score: {micro_f1:.4f}")
print(f"Mean Average Precision (mAP): {mean_ap:.4f}")



#######Q2

# Remove stopwords
def remove_stopwords(doc):
    text = ' '.join([word for word in doc.split() if word.lower() not in stop_words])

def preprocess_document(doc):
    # Flatten dictionary and combine relevant text fields
    text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('full_text', '')}"
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    
    return text

def preprocess_documents(documents):
    return [preprocess_document(doc) for doc in documents]

preprocessed_documents = preprocess_documents(anthology_sample)



# Initialize lists to store embeddings
# Initialize the MiniLM model for document embeddings
# minilm_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Function to get document embedding using MiniLM
# def get_document_embedding(document):
#     return minilm_model.encode(document, convert_to_tensor=False)

# minilm_embeddings = []


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to get document embedding using MiniLM
# def get_document_embedding(document):
#     return minilm_model.encode(document, convert_to_tensor=False)

# Function to get word embeddings using BERT and then aggregate them
def get_aggregated_word_embeddings(document):
    inputs = bert_tokenizer(document, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    word_embeddings = outputs.last_hidden_state.squeeze(0)  # Removing the batch dimension
    aggregated_embedding = word_embeddings.mean(dim=0)  # Mean aggregation
    return aggregated_embedding.numpy()




bert_embeddings = []
glove_embeddings = []

# Generate embeddings for each word
def get_glove_word_embeddings(document):
    word_embeddings = [glove_model[word] for word in document.split() if word in glove_model]
    if not word_embeddings:
        # raise ValueError("None of the words in the input text are in the GloVe model.")

        print("None of the words in the input text are in the GloVe model.")
        print("document:",document)

        zero_array = np.zeros(100, dtype=float) ### document didn't match any words from glove, hence returning zero array
        document_embedding = zero_array
    else:
        document_embedding = np.mean(word_embeddings, axis=0)
        # print("len:", len(document_embedding))
    return document_embedding

# Compute embeddings for the first 1000 documents
count = 1
for document in preprocessed_documents[:]:
    if count % 100 == 0:
        print("Processing doc:",count)
    # Ensure the document is a string
    document = str(document)
    bert_embeddings.append(get_aggregated_word_embeddings(document))
    we = get_glove_word_embeddings(document) ### HAD TO implement this check with we (word embeddings) as some of the documents had 0 hits on glove
    glove_embeddings.append(we)
    count+=1
    
    
# Get embedding for the query sentence
test_query = queries["queries"][50]['q']
test_query_bert_embed = get_aggregated_word_embeddings(test_query).reshape(1, -1)
test_query_glove_embed = get_glove_word_embeddings(test_query).reshape(1, -1)


distances_bert, indices_bert = nn_bert.kneighbors(test_query_bert_embed)
# Print the indices and distances of the nearest neighbors
print("BERT Indices of nearest neighbors:", indices_bert)
print("BERT Distances to nearest neighbors:", distances_bert)

distances_glove, indices_glove = nn_glove.kneighbors(test_query_glove_embed)

# Print the indices and distances of the nearest neighbors
print("Glove Indices of nearest neighbors:", indices_glove)
print("Glove Distances to nearest neighbors:", distances_glove)

# Print the nearest neighbor sentences
# nearest_neighbors = [preprocessed_documents[idx] for idx in indices_glove[0]]
# print("Nearest neighbor sentences:", nearest_neighbors)
# query_embedding2 = get_mpnet_embedding(test_query).reshape(1, -1)
# distances, indices = knn2.kneighbors(query_embedding2)


##########Q3

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import faiss
import numpy as np
import time

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example documents with structure
documents = [
    {"id": "doc1", "title": "Title 1", "abstract": "Abstract 1", "full_text": "Sentence 1 of doc 1. Sentence 2 of doc 1."},
    {"id": "doc2", "title": "Title 2", "abstract": "Abstract 2", "full_text": "Sentence 1 of doc 2. Sentence 2 of doc 2."},
    {"id": "doc3", "title": "Title 3", "abstract": "Abstract 3", "full_text": "Sentence 1 of doc 3. Sentence 2 of doc 3."},
    {"id": "doc4", "title": "Title 4", "abstract": "Abstract 4", "full_text": "Sentence 1 of doc 4. Sentence 2 of doc 4."}
]

# Split full_text into sentences and create a mapping to document ids
all_sentences = []
sentence_to_doc_map = []

# for doc in documents:
#     sentences = doc["full_text"].split(". ")  # Assuming sentences are separated by ". "
#     for sentence in sentences:
#         all_sentences.append(sentence)
#         sentence_to_doc_map.append(doc["id"])

# Create embeddings
embeddings = model.encode(preprocessed_documents)
print("Embeddings created.")


# Initialize KNN
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
# Fit the model
knn.fit(embeddings)



# Perform KNN search
nn = 5
start_time = time.time()
distances, knn_indices = knn.kneighbors(query_embedding,n_neighbors=nn)
knn_time = time.time() - start_time


print("KNN Results:")
for index in knn_indices[0]:
    print(f"Document ID: {preprocess_documents_map[index]}, Sentence: {preprocessed_documents[index]}, Distance: {distances[0][knn_indices[0].tolist().index(index)]}")
print(f"KNN search time: {knn_time} seconds")

# Initialize FAISS index
embedding_dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Add embeddings to FAISS index
faiss_index.add(embeddings)
print("FAISS index built.")

# Perform FAISS search
start_time = time.time()
D, I = faiss_index.search(query_embedding, nn)
faiss_time = time.time() - start_time

print("FAISS Results:")
for i, index in enumerate(I[0]):
    print(f"Document ID: {preprocess_documents_map[index]}, Sentence: {preprocessed_documents[index]}, Distance: {D[0][i]}")
print(f"FAISS search time: {faiss_time} seconds")



######### Q4

import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def extractive_summary(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return ' '.join(str(sentence) for sentence in summary)
    
def preprocess_document(doc):
    # Flatten dictionary and combine relevant text fields
    summary = extractive_summary(doc.get('full_text', ''))
    text = f"{doc.get('title', '')} {doc.get('abstract', '')} {summary}"
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    
    return text


def preprocess_documents(documents):
    return [preprocess_document(doc) for doc in documents]

preprocessed_documents = preprocess_documents(anthology_sample)



# Initialize lists to store embeddings
# Initialize the MiniLM model for document embeddings
minilm_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Function to get document embedding using MiniLM
def get_document_embedding(document):
    return minilm_model.encode(document, convert_to_tensor=False)

minilm_embeddings = []
# bert_embeddings = []

# Compute embeddings for the first 1000 documents
count = 1
for document in preprocessed_documents[:]:
    print("Processing doc:",count)
    # Ensure the document is a string
    document = str(document)
    # print("Processing sentence embeddings:",count)
    minilm_embeddings.append(get_document_embedding(document))
    # print("Processing word embeddings:",count)
    # bert_embeddings.append(get_aggregated_word_embeddings(document))
    count+=1
    

# preprocessed_documents = preprocess_documents(anthology_sample)
# minilm_embeddings2 = [get_mpnet_embedding(doc) for doc in preprocessed_documents[:10]]  # Only first 1000 for k-NN
from sklearn.neighbors import NearestNeighbors
# knn2 = NearestNeighbors(n_neighbors=5, metric='cosine').fit(minilm_embeddings)

# Fit NearestNeighbors model
nn_minilm = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
nn_minilm.fit(minilm_embeddings)


# Get embedding for the query sentence
test_query = queries["queries"][50]['q']
test_query_sen_embed = get_document_embedding(test_query).reshape(1, -1)
# get_nearest_neighbors(test_query_sen_embed,nn_minilm,1)
# get_document_embedding

distances, indices = nn_minilm.kneighbors(test_query_sen_embed)

# Print the indices and distances of the nearest neighbors
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)

# Print the nearest neighbor sentences
nearest_neighbors = [preprocessed_documents[idx] for idx in indices[0]]
# print("Nearest neighbor sentences:", nearest_neighbors)
# query_embedding2 = get_mpnet_embedding(test_query).reshape(1, -1)
# distances, indices = knn2.kneighbors(query_embedding2)

import numpy as np

## Retrieving the query
# Function to retrieve the ground truth for a given query
def get_ground_truth(query):
    for q in queries["queries"]:
        if q["q"] == query:
            # return q["r"]
            return [r for r in q["r"]]
    return None
# ground_truth = get_ground_truth(query)

# Function to compare the result with the ground truth for a single query
# Function to calculate average precision for a single query
def average_precision(retrieved_docs, ground_truth_ids):
    if not ground_truth_ids:
        return 0
    retrieved_docs_set = set(retrieved_docs)
    ground_truth_set = set(ground_truth_ids)
    
    num_relevant = 0
    precision_sum = 0
    for i, doc in enumerate(retrieved_docs):
        if doc in ground_truth_set:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    
    return precision_sum / len(ground_truth_set)

# Function to compare the result with the ground truth for a single query
def compare_with_ground_truth(query_text, nn_model, dataset, ground_truth_function, k):
    # print("cgt k=",k)
    # Get nearest neighbors
    indices = get_nearest_neighbors(query_text, nn_model, k)
    
    # Convert numpy.int64 to Python int
    indices = [int(i) for i in indices]
    
    # Retrieve document IDs for nearest neighbors
    retrieved_docs = [dataset[i]['acl_id'] for i in indices]
    
    # Get ground truth
    ground_truth_ids = ground_truth_function(query_text)
    
    if not ground_truth_ids:
        return 0, 0, 0, 0, 0, 0, 0  # Return zeros if no ground truth is available
    
    # Calculate true positives, false positives, and false negatives
    tp = len(set(retrieved_docs) & set(ground_truth_ids))
    fp = len(retrieved_docs) - tp
    fn = len(ground_truth_ids) - tp
    
    # Calculate precision, recall, and F1-score
    precision = tp / len(retrieved_docs) if retrieved_docs else 0
    recall = tp / len(ground_truth_ids) if ground_truth_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    # Calculate average precision
    ap = average_precision(retrieved_docs, ground_truth_ids)
    
    return precision, recall, f1, ap, tp, fp, fn

# Function to evaluate the model on all queries
def evaluate_model_on_all_queries(queries, embeddings, dataset, ground_truth_function, k):
    # Fit NearestNeighbors model
    print("k=",k)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(embeddings)

    precision_scores = []
    recall_scores = []
    f1_scores = []
    ap_scores = []
    total_tp = total_fp = total_fn = 0

    for query in queries["queries"]:
        query_text = query["q"]
        precision, recall, f1, ap, tp, fp, fn = compare_with_ground_truth(query_text, nn, dataset, ground_truth_function, k)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ap_scores.append(ap)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate macro average metrics
    macro_precision = np.mean(precision_scores)
    macro_recall = np.mean(recall_scores)
    macro_f1 = np.mean(f1_scores)
    
    # Calculate micro average metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0
    
    # Calculate mean AP
    mean_ap = np.mean(ap_scores)
    
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap

def get_nearest_neighbors(query_text, nn_model, k):
    # query_embedding = get_aggregated_word_embeddings([query_text])
    # print(query_text)
    # test_query_word_embed = get_aggregated_word_embeddings(query_text).reshape(1, -1)
    test_query_word_embed = get_document_embedding(query_text).reshape(1, -1)
    distances, indices = nn_model.kneighbors(test_query_word_embed, n_neighbors=k)
    return indices[0]

def nearest_neighbour(embeddings): # can be tfidf_matrix or LSI_matrix doesnt
    nn = NearestNeighbors(n_neighbors=1, metric='cosine',algorithm = 'brute')
    nn.fit(embeddings)
    
# test_query = queries["queries"][0]['q']
# get_nearest_neighbors(test_query,nn_minilm,1)

k = 5
macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap = evaluate_model_on_all_queries(queries, minilm_embeddings, anthology_sample, get_ground_truth, k)
print(f"Macro Average Precision: {macro_precision:.4f}")
print(f"Macro Average Recall: {macro_recall:.4f}")
print(f"Macro Average F1-Score: {macro_f1:.4f}")
print(f"Micro Average Precision: {micro_precision:.4f}")
print(f"Micro Average Recall: {micro_recall:.4f}")
print(f"Micro Average F1-Score: {micro_f1:.4f}")
print(f"Mean Average Precision (mAP): {mean_ap:.4f}")


############## Q6+Q7

import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import datasets


# R_NUMBER_SEED = 1234567 # Replace this with your own student number
R_NUMBER_SEED = 928036 # my student number
DOCS_TO_ADD = 1000
query_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_queries.parquet")["train"]
all_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_full.parquet")["train"]
# Shuffle with seed and take only n docs
shuffled_documents = all_documents.shuffle(seed=R_NUMBER_SEED)
random_documents = shuffled_documents.select(range(DOCS_TO_ADD))
# Concatenate relevant documents with random sample and shuffle again
anthology_sample = datasets.concatenate_datasets([query_documents, random_documents]).shuffle(seed=R_NUMBER_SEED)
# Export to Parquet to avoid downloading full anthology
anthology_sample.to_parquet("./anthology_sample.parquet")


import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def preprocess_documents_nltk(documents):
    all_sentences = []
    all_sentences_to_doc_map = []
    count = 1
    for doc in documents:
        if count % 10 == 0:
            print("doc:",count)
        # print(doc["acl_id"])
        # full_text = f"{doc.get('full_text')}"
        full_text = ""
        if doc["title"] is not None: full_text = f'{doc["title"]}.'
        if doc["abstract"] is not None: full_text = f'{full_text} {doc["abstract"]}.'
        if doc["full_text"] is not None: full_text = f'{full_text} {doc["full_text"]}.'
        sentences = sent_tokenize(full_text)
        for sentence in sentences:
            all_sentences.append(sentence)
            all_sentences_to_doc_map.append(doc["acl_id"])
        count += 1
    return all_sentences,all_sentences_to_doc_map
preprocessed_documents, sentence_to_doc_map = preprocess_documents_nltk(anthology_sample)

# Get embedding for the query sentence
test_query = queries["queries"][18]['q']
test_query_sen_embed = get_document_embedding(test_query).reshape(1, -1)
# get_nearest_neighbors(test_query_sen_embed,nn_minilm,1)
# get_document_embedding

distances, indices = nn_minilm.kneighbors(test_query_sen_embed)

# Print the indices and distances of the nearest neighbors
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)

# Print the nearest neighbor sentences
nearest_neighbors = [preprocessed_documents[idx] for idx in indices[0]]
# print("Nearest neighbor sentences:", nearest_neighbors)
# query_embedding2 = get_mpnet_embedding(test_query).reshape(1, -1)
# distances, indices = knn2.kneighbors(query_embedding2)

import numpy as np

## Retrieving the query
# Function to retrieve the ground truth for a given query
def get_ground_truth(query):
    for q in queries["queries"]:
        if q["q"] == query:
            # return q["r"]
            return [r for r in q["r"]]
    return None
# ground_truth = get_ground_truth(query)

# Function to compare the result with the ground truth for a single query
# Function to calculate average precision for a single query
def average_precision(retrieved_docs, ground_truth_ids):
    if not ground_truth_ids:
        return 0
    retrieved_docs_set = set(retrieved_docs)
    ground_truth_set = set(ground_truth_ids)
    
    num_relevant = 0
    precision_sum = 0
    for i, doc in enumerate(retrieved_docs):
        if doc in ground_truth_set:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    
    return precision_sum / len(ground_truth_set)

# Function to compare the result with the ground truth for a single query
def compare_with_ground_truth(query_text, nn_model, dataset, ground_truth_function, k):
    # print("cgt k=",k)
    # Get nearest neighbors
    indices = get_nearest_neighbors(query_text, nn_model, k)
    
    # Convert numpy.int64 to Python int
    indices = [int(i) for i in indices]
    
    # Retrieve document IDs for nearest neighbors
    retrieved_docs = [dataset[i]['acl_id'] for i in indices]
    
    # Get ground truth
    ground_truth_ids = ground_truth_function(query_text)
    
    if not ground_truth_ids:
        return 0, 0, 0, 0, 0, 0, 0  # Return zeros if no ground truth is available
    
    # Calculate true positives, false positives, and false negatives
    tp = len(set(retrieved_docs) & set(ground_truth_ids))
    fp = len(retrieved_docs) - tp
    fn = len(ground_truth_ids) - tp
    
    # Calculate precision, recall, and F1-score
    precision = tp / len(retrieved_docs) if retrieved_docs else 0
    recall = tp / len(ground_truth_ids) if ground_truth_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    # Calculate average precision
    ap = average_precision(retrieved_docs, ground_truth_ids)
    
    return precision, recall, f1, ap, tp, fp, fn

# Function to evaluate the model on all queries
def evaluate_model_on_all_queries(queries, embeddings, dataset, ground_truth_function, k):
    # Fit NearestNeighbors model
    print("k=",k)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(embeddings)

    precision_scores = []
    recall_scores = []
    f1_scores = []
    ap_scores = []
    total_tp = total_fp = total_fn = 0

    for query in queries["queries"]:
        query_text = query["q"]
        precision, recall, f1, ap, tp, fp, fn = compare_with_ground_truth(query_text, nn, dataset, ground_truth_function, k)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ap_scores.append(ap)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate macro average metrics
    macro_precision = np.mean(precision_scores)
    macro_recall = np.mean(recall_scores)
    macro_f1 = np.mean(f1_scores)
    
    # Calculate micro average metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0
    
    # Calculate mean AP
    mean_ap = np.mean(ap_scores)
    
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap

def get_nearest_neighbors(query_text, nn_model, k):
    # query_embedding = get_aggregated_word_embeddings([query_text])
    # print(query_text)
    # test_query_word_embed = get_aggregated_word_embeddings(query_text).reshape(1, -1)
    test_query_word_embed = get_document_embedding(query_text).reshape(1, -1)
    distances, indices = nn_model.kneighbors(test_query_word_embed, n_neighbors=k)
    return indices[0]

def nearest_neighbour(embeddings): # can be tfidf_matrix or LSI_matrix doesnt
    nn = NearestNeighbors(n_neighbors=1, metric='cosine',algorithm = 'brute')
    nn.fit(embeddings)
    
# test_query = queries["queries"][0]['q']
# get_nearest_neighbors(test_query,nn_minilm,1)


k = 5
macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap = evaluate_model_on_all_queries(queries, minilm_embeddings, anthology_sample, get_ground_truth, k)
print(f"Macro Average Precision: {macro_precision:.4f}")
print(f"Macro Average Recall: {macro_recall:.4f}")
print(f"Macro Average F1-Score: {macro_f1:.4f}")
print(f"Micro Average Precision: {micro_precision:.4f}")
print(f"Micro Average Recall: {micro_recall:.4f}")
print(f"Micro Average F1-Score: {micro_f1:.4f}")
print(f"Mean Average Precision (mAP): {mean_ap:.4f}")



############ Q8


# preprocessed_documents = preprocess_documents(anthology_sample)
# minilm_embeddings2 = [get_mpnet_embedding(doc) for doc in preprocessed_documents[:10]]  # Only first 1000 for k-NN
from sklearn.neighbors import NearestNeighbors
# knn2 = NearestNeighbors(n_neighbors=5, metric='cosine').fit(minilm_embeddings)

# Fit NearestNeighbors model
nn_minilm = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
nn_minilm.fit(minilm_embeddings)


# Get embedding for the query sentence
test_query = queries["queries"][18]['q']
test_query_sen_embed = get_document_embedding(test_query).reshape(1, -1)
# get_nearest_neighbors(test_query_sen_embed,nn_minilm,1)
# get_document_embedding

distances, indices = nn_minilm.kneighbors(test_query_sen_embed)

# Print the indices and distances of the nearest neighbors
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)

# Print the nearest neighbor sentences
nearest_neighbors = [preprocessed_documents[idx] for idx in indices[0]]
# print("Nearest neighbor sentences:", nearest_neighbors)
# query_embedding2 = get_mpnet_embedding(test_query).reshape(1, -1)
# distances, indices = knn2.kneighbors(query_embedding2)



import numpy as np

## Retrieving the query
# Function to retrieve the ground truth for a given query
def get_ground_truth(query):
    for q in queries["queries"]:
        if q["q"] == query:
            # return q["r"]
            return [r for r in q["r"]]
    return None
# ground_truth = get_ground_truth(query)

# Function to compare the result with the ground truth for a single query
# Function to calculate average precision for a single query
def average_precision(retrieved_docs, ground_truth_ids):
    if not ground_truth_ids:
        return 0
    retrieved_docs_set = set(retrieved_docs)
    ground_truth_set = set(ground_truth_ids)
    
    num_relevant = 0
    precision_sum = 0
    for i, doc in enumerate(retrieved_docs):
        if doc in ground_truth_set:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    
    return precision_sum / len(ground_truth_set)

# Function to compare the result with the ground truth for a single query
def compare_with_ground_truth(query_text, nn_model, dataset, ground_truth_function, k):
    # print("cgt k=",k)
    # Get nearest neighbors
    indices = get_nearest_neighbors(query_text, nn_model, k)
    
    # Convert numpy.int64 to Python int
    indices = [int(i) for i in indices]
    
    # Retrieve document IDs for nearest neighbors
    retrieved_docs = [dataset[i]['acl_id'] for i in indices]
    
    # Get ground truth
    ground_truth_ids = ground_truth_function(query_text)
    
    if not ground_truth_ids:
        return 0, 0, 0, 0, 0, 0, 0  # Return zeros if no ground truth is available
    
    # Calculate true positives, false positives, and false negatives
    tp = len(set(retrieved_docs) & set(ground_truth_ids))
    fp = len(retrieved_docs) - tp
    fn = len(ground_truth_ids) - tp
    
    # Calculate precision, recall, and F1-score
    precision = tp / len(retrieved_docs) if retrieved_docs else 0
    recall = tp / len(ground_truth_ids) if ground_truth_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    # Calculate average precision
    ap = average_precision(retrieved_docs, ground_truth_ids)
    
    return precision, recall, f1, ap, tp, fp, fn

# Function to evaluate the model on all queries
def evaluate_model_on_all_queries(queries, embeddings, dataset, ground_truth_function, k):
    # Fit NearestNeighbors model
    print("k=",k)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(embeddings)

    precision_scores = []
    recall_scores = []
    f1_scores = []
    ap_scores = []
    total_tp = total_fp = total_fn = 0

    for query in queries["queries"]:
        query_text = query["q"]
        precision, recall, f1, ap, tp, fp, fn = compare_with_ground_truth(query_text, nn, dataset, ground_truth_function, k)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ap_scores.append(ap)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate macro average metrics
    macro_precision = np.mean(precision_scores)
    macro_recall = np.mean(recall_scores)
    macro_f1 = np.mean(f1_scores)
    
    # Calculate micro average metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0
    
    # Calculate mean AP
    mean_ap = np.mean(ap_scores)
    
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap







def get_nearest_neighbors(query_text, nn_model, k):
    # query_embedding = get_aggregated_word_embeddings([query_text])
    # print(query_text)
    # test_query_word_embed = get_aggregated_word_embeddings(query_text).reshape(1, -1)
    test_query_word_embed = get_document_embedding(query_text).reshape(1, -1)
    distances, indices = nn_model.kneighbors(test_query_word_embed, n_neighbors=k)
    return indices[0]

def nearest_neighbour(embeddings): # can be tfidf_matrix or LSI_matrix doesnt
    nn = NearestNeighbors(n_neighbors=1, metric='cosine',algorithm = 'brute')
    nn.fit(embeddings)
    
# test_query = queries["queries"][0]['q']
# get_nearest_neighbors(test_query,nn_minilm,1)

k = 5
macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap = evaluate_model_on_all_queries(queries, minilm_embeddings, anthology_sample, get_ground_truth, k)
print(f"Macro Average Precision: {macro_precision:.4f}")
print(f"Macro Average Recall: {macro_recall:.4f}")
print(f"Macro Average F1-Score: {macro_f1:.4f}")
print(f"Micro Average Precision: {micro_precision:.4f}")
print(f"Micro Average Recall: {micro_recall:.4f}")
print(f"Micro Average F1-Score: {micro_f1:.4f}")
print(f"Mean Average Precision (mAP): {mean_ap:.4f}")

####### to make run OpenAI API for the questions output

import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import datasets


# R_NUMBER_SEED = 1234567 # Replace this with your own student number
R_NUMBER_SEED = 928036 # my student number
DOCS_TO_ADD = 1000
query_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_queries.parquet")["train"]
all_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_full.parquet")["train"]
# Shuffle with seed and take only n docs
shuffled_documents = all_documents.shuffle(seed=R_NUMBER_SEED)
random_documents = shuffled_documents.select(range(DOCS_TO_ADD))
# Concatenate relevant documents with random sample and shuffle again
anthology_sample = datasets.concatenate_datasets([query_documents, random_documents]).shuffle(seed=R_NUMBER_SEED)
# Export to Parquet to avoid downloading full anthology
anthology_sample.to_parquet("./anthology_sample.parquet")

import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import datasets


# R_NUMBER_SEED = 1234567 # Replace this with your own student number
R_NUMBER_SEED = 928036 # my student number
DOCS_TO_ADD = 1000
query_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_queries.parquet")["train"]
all_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_full.parquet")["train"]
# Shuffle with seed and take only n docs
shuffled_documents = all_documents.shuffle(seed=R_NUMBER_SEED)
random_documents = shuffled_documents.select(range(DOCS_TO_ADD))
# Concatenate relevant documents with random sample and shuffle again
anthology_sample = datasets.concatenate_datasets([query_documents, random_documents]).shuffle(seed=R_NUMBER_SEED)
# Export to Parquet to avoid downloading full anthology
anthology_sample.to_parquet("./anthology_sample.parquet")

# Initialize lists to store embeddings
# Initialize the MiniLM model for document embeddings
minilm_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Function to get document embedding using MiniLM
def get_document_embedding(document):
    return minilm_model.encode(document, convert_to_tensor=False)

minilm_embeddings = []
# bert_embeddings = []

# Compute embeddings for the first 1000 documents
indexes = []
count = 1
for document in preprocessed_documents[:10000]:
    if count % 100 == 0:
        print("Processing doc:",count)
    # Ensure the document is a string
    document = str(document)
    # print("Processing sentence embeddings:",count)
    indexes.append(document["acl_id"])
    count+=1
    
# preprocessed_documents = preprocess_documents(anthology_sample)
# minilm_embeddings2 = [get_mpnet_embedding(doc) for doc in preprocessed_documents[:10]]  # Only first 1000 for k-NN
from sklearn.neighbors import NearestNeighbors
# knn2 = NearestNeighbors(n_neighbors=5, metric='cosine').fit(minilm_embeddings)

# Fit NearestNeighbors model
nn_minilm = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
nn_minilm.fit(minilm_embeddings)


# Get embedding for the query sentence
test_query = queries["queries"][18]['q']
test_query_sen_embed = get_document_embedding(test_query).reshape(1, -1)
# get_nearest_neighbors(test_query_sen_embed,nn_minilm,1)
# get_document_embedding

distances, indices = nn_minilm.kneighbors(test_query_sen_embed)

# Print the indices and distances of the nearest neighbors
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)

# Print the nearest neighbor sentences
nearest_neighbors = [preprocessed_documents[idx] for idx in indices[0]]
# print("Nearest neighbor sentences:", nearest_neighbors)
# query_embedding2 = get_mpnet_embedding(test_query).reshape(1, -1)
# distances, indices = knn2.kneighbors(query_embedding2)


import numpy as np

## Retrieving the query
# Function to retrieve the ground truth for a given query
def get_ground_truth(query):
    for q in queries["queries"]:
        if q["q"] == query:
            # return q["r"]
            return [r for r in q["r"]]
    return None
# ground_truth = get_ground_truth(query)

# Function to compare the result with the ground truth for a single query
# Function to calculate average precision for a single query
def average_precision(retrieved_docs, ground_truth_ids):
    if not ground_truth_ids:
        return 0
    retrieved_docs_set = set(retrieved_docs)
    ground_truth_set = set(ground_truth_ids)
    
    num_relevant = 0
    precision_sum = 0
    for i, doc in enumerate(retrieved_docs):
        if doc in ground_truth_set:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    
    return precision_sum / len(ground_truth_set)

# Function to compare the result with the ground truth for a single query
def compare_with_ground_truth(query_text, nn_model, dataset, ground_truth_function, k):
    # print("cgt k=",k)
    # Get nearest neighbors
    indices = get_nearest_neighbors(query_text, nn_model, k)
    
    # Convert numpy.int64 to Python int
    indices = [int(i) for i in indices]
    
    # Retrieve document IDs for nearest neighbors
    retrieved_docs = [dataset[i]['acl_id'] for i in indices]
    
    # Get ground truth
    ground_truth_ids = ground_truth_function(query_text)
    
    if not ground_truth_ids:
        return 0, 0, 0, 0, 0, 0, 0  # Return zeros if no ground truth is available
    
    # Calculate true positives, false positives, and false negatives
    tp = len(set(retrieved_docs) & set(ground_truth_ids))
    fp = len(retrieved_docs) - tp
    fn = len(ground_truth_ids) - tp
    
    # Calculate precision, recall, and F1-score
    precision = tp / len(retrieved_docs) if retrieved_docs else 0
    recall = tp / len(ground_truth_ids) if ground_truth_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    # Calculate average precision
    ap = average_precision(retrieved_docs, ground_truth_ids)
    
    return precision, recall, f1, ap, tp, fp, fn

# Function to evaluate the model on all queries
def evaluate_model_on_all_queries(queries, embeddings, dataset, ground_truth_function, k):
    # Fit NearestNeighbors model
    print("k=",k)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(embeddings)

    precision_scores = []
    recall_scores = []
    f1_scores = []
    ap_scores = []
    total_tp = total_fp = total_fn = 0

    for query in queries["queries"]:
        query_text = query["q"]
        precision, recall, f1, ap, tp, fp, fn = compare_with_ground_truth(query_text, nn, dataset, ground_truth_function, k)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ap_scores.append(ap)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate macro average metrics
    macro_precision = np.mean(precision_scores)
    macro_recall = np.mean(recall_scores)
    macro_f1 = np.mean(f1_scores)
    
    # Calculate micro average metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0
    
    # Calculate mean AP
    mean_ap = np.mean(ap_scores)
    
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap

def get_nearest_neighbors(query_text, nn_model, k):
    # query_embedding = get_aggregated_word_embeddings([query_text])
    # print(query_text)
    # test_query_word_embed = get_aggregated_word_embeddings(query_text).reshape(1, -1)
    test_query_word_embed = get_document_embedding(query_text).reshape(1, -1)
    distances, indices = nn_model.kneighbors(test_query_word_embed, n_neighbors=k)
    return indices[0]

def nearest_neighbour(embeddings): # can be tfidf_matrix or LSI_matrix doesnt
    nn = NearestNeighbors(n_neighbors=1, metric='cosine',algorithm = 'brute')
    nn.fit(embeddings)
    
# test_query = queries["queries"][0]['q']
# get_nearest_neighbors(test_query,nn_minilm,1)



k = 5
macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, mean_ap = evaluate_model_on_all_queries(queries, minilm_embeddings, anthology_sample, get_ground_truth, k)
print(f"Macro Average Precision: {macro_precision:.4f}")
print(f"Macro Average Recall: {macro_recall:.4f}")
print(f"Macro Average F1-Score: {macro_f1:.4f}")
print(f"Micro Average Precision: {micro_precision:.4f}")
print(f"Micro Average Recall: {micro_recall:.4f}")
print(f"Micro Average F1-Score: {micro_f1:.4f}")
print(f"Mean Average Precision (mAP): {mean_ap:.4f}")





























"""
