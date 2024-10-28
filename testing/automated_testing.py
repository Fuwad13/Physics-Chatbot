import json
import requests
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

MODEL_LIST = ["llama3.2:latest", "llama3.1:latest", "mistral:latest", "gemma2:latest"] 
EMBEDDING_MODEL_LIST = ["nomic-embed-text", "chroma/all-minilm-l6-v2-f32", "mxbai-embed-large"]

API_BASE_URL = "http://localhost:8080"


def calculate_bleu_score(dataset_file: str):
    bleu_scores = []
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    
    bleu_scores = []
    # Iterate over each question-answer pair
    for data in dataset:
        question = data["question"]
        ideal_answer = data["ideal_answer"]

        # Get chatbot's answer
        payload = {
                "question": question,
                "llm_model": MODEL_LIST[0], # llama3.2:latest
                "embedding_model": EMBEDDING_MODEL_LIST[0] # nomic-embed-text
            }
        
        response = requests.post(f"{API_BASE_URL}/ask_question", json=payload)
        response = response.json()
        chatbot_answer = response["model_response"]

        # Calculate BLEU score
        reference = [ideal_answer.split()]  # List of reference words
        hypothesis = chatbot_answer.split()  # Tokenized chatbot answer

        bleu_score = sentence_bleu(reference, hypothesis)
        bleu_scores.append(bleu_score)

        average_bleu = sum(bleu_scores) / len(bleu_scores)
        print("Average BLEU score:", average_bleu)
        return average_bleu

def calculate_metrics(dataset_file: str):
    # Initialize lists to store scores
    bleu_scores = []
    meteor_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    # Load dataset
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    
    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Iterate over each question-answer pair
    for data in dataset:
        question = data["question"]
        ideal_answer = data["ideal_answer"]

        # Get chatbot's answer
        payload = {
            "question": question,
            "llm_model": MODEL_LIST[0],  # llama3.2:latest
            "embedding_model": EMBEDDING_MODEL_LIST[0]  # nomic-embed-text
        }
        
        response = requests.post(f"{API_BASE_URL}/ask_question", json=payload)
        response = response.json()
        chatbot_answer = response["model_response"]

        # Tokenize reference and hypothesis for BLEU and ROUGE
        reference = [ideal_answer.split()]
        hypothesis = chatbot_answer.split()

        # Calculate BLEU score
        bleu_score = sentence_bleu(reference, hypothesis)
        bleu_scores.append(bleu_score)

        # Calculate METEOR score
        meteor = meteor_score(reference, hypothesis)
        meteor_scores.append(meteor)

        # Calculate ROUGE scores
        rouge_scores_dict = rouge_scorer_obj.score(ideal_answer, chatbot_answer)
        rouge_scores['rouge1'].append(rouge_scores_dict['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(rouge_scores_dict['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(rouge_scores_dict['rougeL'].fmeasure)
    
    # Calculate averages for each metric
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    average_meteor = sum(meteor_scores) / len(meteor_scores)
    average_rouge = {key: sum(values) / len(values) for key, values in rouge_scores.items()}

    print("Average BLEU score:", average_bleu)
    print("Average METEOR score:", average_meteor)
    print("Average ROUGE scores:", average_rouge)
    
    return {
        "average_bleu": average_bleu,
        "average_meteor": average_meteor,
        "average_rouge": average_rouge
    }

# calculate_metrics("E:\\Physics-Chatbot\\testing\\dataset.json")
# calculate_bleu_score("E:\\Physics-Chatbot\\testing\\dataset.json")
    
    
    










