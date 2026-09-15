import asyncio
import json
import os
import re
from typing import List, Tuple, Dict, Any

import importlib

# Import dynamically so environments without the optional Gemini package can
# still parse this module without an editor import-resolution error.
genai = importlib.import_module("google.generativeai")
import pdfplumber
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from dotenv import load_dotenv
import spacy

_ = load_dotenv()

# Configure Gemini API
model = genai.GenerativeModel("gemini-2.0-flash")
genai.configure(api_key=os.getenv("GEMINIAPIKEY"))

# Load NLP models - Using SpaCy instead of BERT
try:
    nlp = spacy.load("en_core_web_sm")
    keyword_model = KeyBERT()
    print("NLP models loaded successfully")
except Exception as e:
    print(f"Error loading NLP models: {e}")
    print("Installing required NLP models...")
    os.system("python -m pip install keybert spacy")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    keyword_model = KeyBERT()

# PDF Text Extraction Functions
async def extract(pdf_path: str) -> str:
    """Extracts text from a PDF asynchronously."""
    try:
        return await asyncio.to_thread(sync_extract, pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def sync_extract(pdf_path: str) -> str:
    """Synchronous function for extracting text from PDF."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error in sync_extract for {pdf_path}: {e}")
    return text

# Gemini API Functions
def get_prompt(resume_text: str) -> str:
    """Creates a prompt for extracting structured data from resume text."""
    return f"""
    Extract the following structured data from the resume:
    - Name
    - Skills (technical and soft skills)
    - Years of Experience (total and per technology/role)
    - Projects (with technologies used)
    - Previous Companies and Roles
    - Areas of Expertise
    
    Return the response as a clean JSON object. Don't use any markdown in response.
    
    Resume: {resume_text}
    """

async def get_response(resume_text: str) -> Dict[str, Any]:
    """Gets structured data from resume text using Gemini API."""
    try:
        response = await model.generate_content_async(get_prompt(resume_text))
       # print("Received response from Gemini API", flush=True)
        response_text = re.sub(r'```json|```', '', response.text.strip())
        return json.loads(response_text)
    except Exception as e:
      #  print(f"Error getting response from Gemini API: {e}")
        return {}

# SpaCy NER Functions
def extract_entities_spacy(text: str) -> Dict[str, List[str]]:
    """Extract named entities using SpaCy."""
    max_chunk_length = 10000
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    grouped_entities = {"ORG": [], "PERSON": [], "GPE": [], "LOC": [], "PRODUCT": [], "SKILL": []}
    
    for chunk in chunks:
        try:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ in grouped_entities and ent.text not in grouped_entities[ent.label_]:
                    grouped_entities[ent.label_].append(ent.text)
            for nc in doc.noun_chunks:
                if len(nc.text) > 3 and nc.text.lower() not in ['the', 'and', 'with', 'using']:
                    lower_chunk = nc.text.lower()
                    tech_patterns = ['python', 'java', 'c++', 'javascript', 'react', 'node', 'aws', 
                                    'cloud', 'database', 'sql', 'nosql', 'api', 'ml', 'ai', 
                                    'machine learning', 'deep learning', 'analytics']
                    if any(tech in lower_chunk for tech in tech_patterns):
                        if nc.text not in grouped_entities["SKILL"]:
                            grouped_entities["SKILL"].append(nc.text)
        except Exception as e:
            print(f"Error processing chunk with SpaCy: {e}")
    return grouped_entities

# Vectorization and Similarity Functions
def extract_keywords_from_job(job_description: str, top_n: int = 20) -> List[Tuple[str, float]]:
    """Extract important keywords from job description using KeyBert."""
    return keyword_model.extract_keywords(
        job_description, keyphrase_ngram_range=(1, 3), stop_words='english', 
        use_mmr=True, diversity=0.7, top_n=top_n
    )

def create_dynamic_weights(keywords: List[Tuple[str, float]], vectorizer) -> np.ndarray:
    """Create weights based on extracted keywords and TF-IDF feature names."""
    weights = np.ones(len(vectorizer.get_feature_names_out()))
    feature_mapping = {feature: idx for idx, feature in enumerate(vectorizer.get_feature_names_out())}
    for keyword, importance in keywords:
        for feature, idx in feature_mapping.items():
            if keyword.lower() in feature.lower():
                weights[idx] = 3.0 + (importance * 2.0)
    return weights

def enhance_job_description(job_description: str) -> str:
    """Enhance job description by identifying key requirements using SpaCy NER."""
    entities = extract_entities_spacy(job_description)
    technical_terms = entities["ORG"] + entities["PRODUCT"] + entities["SKILL"]
    requirement_indicators = ["required", "must have", "essential", "necessary", "qualification"]
    important_sentences = [sent for sent in re.split(r'(?<=[.!?])\s+', job_description) 
                          if any(ind in sent.lower() for ind in requirement_indicators)]
    enhanced = job_description + "\n\nKEY FOCUS: " + " ".join(technical_terms)
    if important_sentences:
        enhanced += "\n\nCRITICAL REQUIREDMENTS: " + " ".join(important_sentences)
    return enhanced

def convert_to_tfidf_vectors(resume_texts: List[str], job_description: str):
    """Converts resume texts and job description to TF-IDF vectors."""
    vectorizer = TfidfVectorizer(
        max_features=2000, stop_words="english", ngram_range=(1, 3), 
        sublinear_tf=True, min_df=1, max_df=0.85, use_idf=True, norm='l2'
    )
    enhanced_job = enhance_job_description(job_description)
    all_texts = resume_texts + [enhanced_job]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    resume_vectors = tfidf_matrix[:-1]
    job_vector = tfidf_matrix[-1]
    keywords = extract_keywords_from_job(job_description)
    print("\nKey job requirements identified:")
    for keyword, score in keywords[:10]:
        print(f"- {keyword} (importance: {score:.2f})")
    job_entities = extract_entities_spacy(job_description)
    print("\nNamed entities from job description:")
    for entity_type, entities in job_entities.items():
        if entities:
            print(f"- {entity_type}: {', '.join(entities[:5])}")
    weights = create_dynamic_weights(keywords, vectorizer)
    feature_mapping = {feature: idx for idx, feature in enumerate(vectorizer.get_feature_names_out())}
    for entity_type, entities in job_entities.items():
        multiplier = 2.0 if entity_type in ["ORG", "SKILL"] else 1.5
        for entity in entities:
            for feature, idx in feature_mapping.items():
                if entity.lower() in feature.lower():
                    weights[idx] *= multiplier
    joblib.dump(resume_vectors, "resume_vectors.pkl")
    joblib.dump(job_vector, "job_vector.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(weights, "feature_weights.pkl")
    return resume_vectors, job_vector, vectorizer, weights

def extract_entities_from_resumes(resume_texts: List[str]) -> List[Dict[str, List[str]]]:
    """Extract named entities from all resumes using SpaCy NER."""
    return [extract_entities_spacy(text) for text in resume_texts]

def adjust_weights_from_resume_data(weights: np.ndarray, resume_data: List[Dict], 
                                   resume_entities: List[Dict[str, List[str]]], 
                                   vectorizer) -> np.ndarray:
    """Adjust weights based on extracted resume structured data and named entities."""
    adjusted_weights = weights.copy()
    feature_mapping = {feature: idx for idx, feature in enumerate(vectorizer.get_feature_names_out())}
    all_skills = []
    for data in resume_data:
        if data and 'Skills' in data:
            skills = data['Skills']
            if isinstance(skills, list):
                pass
            elif isinstance(skills, str):
                skills = skills.split(',')
            elif isinstance(skills, dict):
                skills = list(skills.keys())
            else:
                skills = []
            all_skills.extend([s.strip() for s in skills])
    all_orgs = []
    all_skills_from_ner = []
    for entities in resume_entities:
        all_orgs.extend(entities.get("ORG", []))
        all_skills_from_ner.extend(entities.get("SKILL", []))
    all_skills.extend(all_skills_from_ner)
    skill_counts = {skill.lower(): all_skills.count(skill.lower()) for skill in all_skills if isinstance(skill, str)}
    org_counts = {org.lower(): all_orgs.count(org.lower()) for org in all_orgs}
    for skill, count in skill_counts.items():
        normalized_count = count / max(len(resume_data), 1)
        for feature, idx in feature_mapping.items():
            if skill in feature.lower():
                adjusted_weights[idx] *= 1.0 + (normalized_count * 2.0)
    for org, count in org_counts.items():
        normalized_count = count / max(len(resume_entities), 1)
        for feature, idx in feature_mapping.items():
            if org in feature.lower():
                adjusted_weights[idx] *= 1.0 + (normalized_count * 1.5)
    return adjusted_weights

def compute_improved_weighted_similarity(resume_vectors, job_vector, candidate_names, resume_data, 
                                        resume_entities, weights=None, vectorizer=None, k=5):
    """Computes improved weighted cosine similarity between job vector and resume vectors."""
    if weights is None or vectorizer is None:
        similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]
    else:
        final_weights = adjust_weights_from_resume_data(weights, resume_data, resume_entities, vectorizer)
        weighted_job_vector = job_vector.multiply(final_weights)
        weighted_resume_vectors = resume_vectors.multiply(final_weights)
        similarity_scores = cosine_similarity(weighted_job_vector, weighted_resume_vectors)[0] 
    top_k_indices = np.argpartition(similarity_scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(-similarity_scores[top_k_indices])]
    top_k_scores = similarity_scores[top_k_indices] * 100 * 1.9
    top_k_names = [candidate_names[i] for i in top_k_indices]
    top_k_results = []
    for i, (name, score) in enumerate(zip(top_k_names, top_k_scores)):
        idx = top_k_indices[i]
        details = []
        if idx < len(resume_data) and resume_data[idx]:
            skills_data = resume_data[idx].get('Skills',[])
            if isinstance(skills_data, str):
                skills = [s.strip() for s in skills_data.split(',')]
            elif isinstance(skills_data, dict):
                skills = list(skills_data.keys())
            else:
                skills = skills_data
            if isinstance(skills, list) and skills:
                key_skills = ", ".join(skills[:3]) if len(skills) > 3 else ", ".join(skills)
                details.append(f"Key skills: {key_skills}")
            experience = resume_data[idx].get('Years of Experience', 'Not specified')
            details.append(f"Experience: {experience}")
        if idx < len(resume_entities) and resume_entities[idx].get('ORG', []):
            details.append(f"Companies: {', '.join(resume_entities[idx]['ORG'][:2])}")
        result = f"{name}: {score:.1f}%"
        if details:
            result += f" - {'. '.join(details)}"
        top_k_results.append(result)
    return top_k_results, similarity_scores

def display_results(top_k_results, similarity_scores, resume_vectors):
    """Displays the matching results with more details."""
    print("\nTop 5 Matching Resumes:")
    for result in top_k_results:
        print(result)
    

# Batch Processing Functions
async def process_pdf_batch(files_batch, semaphore):
    """Process a batch of PDFs with controlled concurrency."""
    async with semaphore:
        return await asyncio.gather(*[extract(file) for file in files_batch])

async def process_gemini_batch(texts_batch, semaphore):
    """Process a batch of texts with Gemini API with controlled concurrency."""
    async with semaphore:
        return await asyncio.gather(*[get_response(text) for text in texts_batch])

# Main Function
async def main():
   
    directory = "./uploads/"
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    pdf_files.sort()
    files = [os.path.join(directory, f) for f in pdf_files]
    #print(files)
    
    # Get job description
    job_description = input("Enter the job description to find the best matching resumes:\n")
    
    # Semaphores for concurrency
    pdf_semaphore = asyncio.Semaphore(10)
    gemini_semaphore = asyncio.Semaphore(5)
    batch_size = 15
    
    # Extract text from PDFs
    print("Extracting text from PDFs...", flush=True)
    extracted_texts = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        batch_texts = await process_pdf_batch(batch, pdf_semaphore)
        extracted_texts.extend(batch_texts)
    
    # Process with Gemini API
    print("Processing resumes with Gemini API...", flush=True)
    resume_data = []
    for i in range(0, len(extracted_texts), batch_size):
        batch = extracted_texts[i:i+batch_size]
        batch_data = await process_gemini_batch(batch, gemini_semaphore)
        resume_data.extend(batch_data)
    
    # Extract entities with SpaCy
    print("Extracting named entities from resumes...", flush=True)
    resume_entities = []
    for i in range(0, len(extracted_texts), batch_size):
        chunk_texts = extracted_texts[i:i+batch_size]
        chunk_entities = await asyncio.to_thread(lambda texts: [extract_entities_spacy(text) for text in texts], chunk_texts)
        resume_entities.extend(chunk_entities)
    
    # Get candidate names
    candidate_names = []
    for i, data in enumerate(resume_data):
        name = data.get('Name', resume_entities[i].get('PERSON', [f"Candidate from {os.path.basename(files[i])}"])[0] if resume_entities[i].get('PERSON') else f"Candidate from {os.path.basename(files[i])}")
        candidate_names.append(name)
    
    # Vectorize
    print("Vectorizing resumes and job description...", flush=True)
    resume_vectors, job_vector, vectorizer, weights = convert_to_tfidf_vectors(extracted_texts, job_description)
    
    # Compute similarity
    print("Computing weighted similarity scores...", flush=True)
    top_k_results, similarity_scores = compute_improved_weighted_similarity(
        resume_vectors, job_vector, candidate_names, resume_data, resume_entities, weights, vectorizer, k=5
    )
    
    # Show results
    display_results(top_k_results, similarity_scores, resume_vectors)

if __name__ == "__main__":
    asyncio.run(main())



