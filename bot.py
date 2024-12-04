import google.generativeai as genai
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

genai.configure(api_key="AIzaSyCq-9E_exJEB7d6hx1822dXmrxnVG2Foyg")

def find_similar_question(user_question, csv_path, threshold=0.6):
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        questions = df['QUESTIONS'].tolist()

        vectorizer = TfidfVectorizer(stop_words='english')
        all_questions = questions + [user_question]
        question_vectors = vectorizer.fit_transform(all_questions)

        cosine_sim = cosine_similarity(question_vectors[-1], question_vectors[:-1])
        max_sim_index = cosine_sim.argmax()
        max_sim_score = cosine_sim[0, max_sim_index]

        if max_sim_score >= threshold:
            return questions[max_sim_index], df.iloc[max_sim_index]["ANSWERS"]
        else:
            return None, None
    except Exception as e:
        print(f"Error in finding similar question: {e}")
        return None, None

def store_qna_to_csv(user_question, answer, csv_path):
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([user_question, answer])

def process_pdf_and_answer_question(pdf_path, output_txt_path, csv_path):

    with open(output_txt_path, "r", encoding="utf-8") as text_file:
        pdf_text = text_file.read()

    while True:
        user_question = input("\nAsk a question based on the PDF (or type 'exit' to quit): ").strip()
        if user_question.lower() == 'exit':
            print("Goodbye!")
            break

        similar_question, answer = find_similar_question(user_question, csv_path, threshold=0.6)

        if similar_question:
            print(f"\nAnswer (from CSV): {answer}")
        else:
            answer = ask_gemini(user_question, pdf_text)
            print(f"\nAnswer (from Gemini): {answer}")
            store_qna_to_csv(user_question, answer, csv_path)

def ask_gemini(question, pdf_text):
    prompt = f"""
The following is a document:
{pdf_text}

Answer the following question based on the document:
{question}
"""
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(f"Context: {pdf_text}\nQuestion: {question}")
    return response.text

pdf_path = "PO_vol1_legislativeEnactments.pdf"
output_txt_path = "extracted_text.txt"
csv_path = "qna.csv"

process_pdf_and_answer_question(pdf_path, output_txt_path, csv_path)



