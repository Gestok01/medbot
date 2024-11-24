from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

healthcare_related_answers = {}
answers_file_path = os.path.join(os.path.dirname(__file__), 'answers.csv')
with open(answers_file_path, 'r') as f:
    reader = csv.reader(f)
    try:
        healthcare_related_answers = dict(reader)
    except ValueError as e:
        print(f"Error: {e}")
        for row in reader:
            print(row)

healthcare_related_questions = []
with open(answers_file_path, 'r') as f:
    reader = csv.reader(f)
    healthcare_related_questions = list(reader)
healthcare_related_questions = [item for sublist in healthcare_related_questions for item in sublist]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(healthcare_related_questions)

def is_healthcare_related(question):
    user_vector = vectorizer.transform([question])
    similarities = cosine_similarity(user_vector, question_vectors).flatten()
    max_similarity_index = similarities.argmax()
    similarity_threshold = 0.5  
    if similarities[max_similarity_index] > similarity_threshold:
        return True
    else:
        return False

def get_answer(question):
    user_vector = vectorizer.transform([question])
    similarities = cosine_similarity(user_vector, question_vectors).flatten()
    max_similarity_index = similarities.argmax()
    similarity_threshold = 0.5  
    if similarities[max_similarity_index] > similarity_threshold:
        matching_question = healthcare_related_questions[max_similarity_index]
        answer = healthcare_related_answers.get(matching_question, None)
        if answer is None:
            return "error"
        else:
            if '"' in answer:
                answer = answer.replace('"', '')
            return answer
    else:
        return "I'm sorry, I don't understand. Please rephrase your question."

def generate_content(question):
    api_key = "AIzaSyBB822ump5y7F06Bdii6lop98bvoQFLF6Y"
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=" + api_key
    headers = {'Content-Type': 'application/json'}

    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": question
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        try:
            response_data = response.json()
            generated_content = response_data.get('candidates', [])[0].get('content', {}).get('parts', [])[0].get('text', '')
            return generated_content
        except:
            return "Sorry, technical error occurred"
    else:
        return "Sorry, technical error occurred"

@app.post("/ask")
def answer_question(request: QuestionRequest):
    question = request.question
    if is_healthcare_related(question):
        return {"answer": get_answer(question)}
    else:
        return {"answer": generate_content(question)}


