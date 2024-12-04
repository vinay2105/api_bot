from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

# Configure Gemini API
api_key = os.getenv("api_key")  # Ensure this is set in the environment
if not api_key:
    raise ValueError("API key for Gemini is not set in the environment")
genai.configure(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

# Define request model for questions
class QuestionRequest(BaseModel):
    user_question: str

# Helper function to find similar questions from the CSV
def find_similar_question(user_question, csv_path, threshold=0.6):
    try:
        # Check if the CSV file exists
        if not os.path.exists(csv_path):
            return None, None

        # Load existing questions and answers
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        if "QUESTIONS" not in df.columns or "ANSWERS" not in df.columns:
            raise ValueError("CSV file must have 'QUESTIONS' and 'ANSWERS' columns")

        questions = df["QUESTIONS"].tolist()

        # Compute cosine similarity
        vectorizer = TfidfVectorizer(stop_words="english")
        all_questions = questions + [user_question]
        question_vectors = vectorizer.fit_transform(all_questions)
        cosine_sim = cosine_similarity(question_vectors[-1], question_vectors[:-1])
        max_sim_index = cosine_sim.argmax()
        max_sim_score = cosine_sim[0, max_sim_index]

        # Return the most similar question if the similarity score meets the threshold
        if max_sim_score >= threshold:
            return questions[max_sim_index], df.iloc[max_sim_index]["ANSWERS"]
        else:
            return None, None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar question: {str(e)}")

# Helper function to store a new question-answer pair in the CSV
def store_qna_to_csv(user_question, answer, csv_path):
    try:
        # Ensure the CSV file exists with proper headers
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["QUESTIONS", "ANSWERS"])

        # Append the new question-answer pair
        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([user_question, answer])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing question and answer: {str(e)}")

# Helper function to query Gemini API
def ask_gemini(question, pdf_text):
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(f"Context: {pdf_text}\nQuestion: {question}")
    return response.text

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to the Q&A API! Use the /ask_question endpoint to ask a question.",
        "example_query": {
            "method": "POST",
            "url": "/ask_question",
            "body": {"user_question": "What is the main topic discussed in the document?"}
        }
    }

# Endpoint to handle user questions
@app.post("/ask_question")
def ask_question(request: QuestionRequest):
    user_question = request.user_question
    csv_path = "qna.csv"

    try:
        # Load the extracted text from the PDF
        with open("extracted_text.txt", "r", encoding="utf-8") as file:
            pdf_text = file.read()

        # Check for a similar question in the CSV
        similar_question, answer = find_similar_question(user_question, csv_path, threshold=0.6)

        if similar_question:
            return {"answer": answer, "source": "csv"}
        else:
            # Query Gemini API for a new answer
            answer = ask_gemini(user_question, pdf_text)
            store_qna_to_csv(user_question, answer, csv_path)
            return {"answer": answer, "source": "gemini"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Extracted text file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

