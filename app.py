from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample Job Description
job_description = """
Looking for a Python developer with knowledge in Flask, Machine Learning,
Data Structures, Algorithms, DBMS, HTML, CSS and SQL.
"""

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_file = request.files['resume']
    
    if resume_file:
        resume_text = extract_text_from_pdf(resume_file)
        
        documents = [resume_text, job_description]
        
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(documents)
        
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        match_percentage = round(float(similarity_score[0][0]) * 100, 2)
        
        return render_template('index.html', match=match_percentage)
    
    return render_template('index.html', match=0)

if __name__ == "__main__":
    app.run(debug=True)
