import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load JSON data
with open("/content/Ecommerce_FAQ_Chatbot_dataset.json", "r") as file:
    faq_data = json.load(file)

# Extract questions and answers
faq_questions = [item['question'] for item in faq_data["questions"]]
faq_answers = [item['answer'] for item in faq_data["questions"]]

# Generate embeddings for FAQ questions
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
faq_embeddings = sentence_model.encode(faq_questions, convert_to_tensor=False)

# Create a FAISS index
dimension = faq_embeddings.shape[1]  # Embedding size
index = faiss.IndexFlatL2(dimension)
index.add(np.array(faq_embeddings))  # Add embeddings to the index

# Save the embeddings to a .npy file
np.save("faq_embeddings.npy", faq_embeddings)

# Save the FAISS index to a file
faiss.write_index(index, "faq_index.index")

print("Embeddings and FAISS index saved successfully.")