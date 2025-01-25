import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import google.generativeai as genai
from transformers import pipeline

# Set the API Key directly in the backend code
api_key = "API_key"  # Replace with your actual API key
genai.configure(api_key=api_key)

# Load pre-trained models from Hugging Face
intent_classifier = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Load precomputed FAQ data and FAISS index
faq_embeddings = np.load("faq_embeddings.npy")
faq_index = faiss.read_index("faq_index.index")

# Load the FAQ data from the original file for answers
with open("Ecommerce_FAQ_Chatbot_dataset.json", "r") as file:
    faq_data = json.load(file)

faq_questions = [item['question'] for item in faq_data["questions"]]
faq_answers = [item['answer'] for item in faq_data["questions"]]

# Initialize the sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit frontend
st.title("NLP-powered Customer Service Assistant")

# Ask user for input (query placeholder)
user_input = st.text_input("Ask me anything:")

# Button to process the query
if st.button("Send"):
    if user_input:
        # Intent Classification
        intent = intent_classifier(user_input, candidate_labels=["Product Search", "FAQ Inquiry", "Order Tracking", "General Chat"])
        st.write(f"Detected Intent: {intent['labels'][0]}")

        # Named Entity Recognition (NER)
        entities = ner_model(user_input)
        st.write(f"Extracted Entities: {entities}")

        # Handling based on Intent
        if intent['labels'][0] == "FAQ Inquiry":
            # Convert user input to embeddings
            query_embedding = sentence_model.encode(user_input, convert_to_tensor=False)

            # Perform similarity search using Faiss
            _, indices = faq_index.search(np.array([query_embedding]), k=1)  # Get top 1 most similar FAQ
            top_idx = indices[0][0]  # Top index of the most similar FAQ

            # Retrieve the most relevant FAQ answer
            faq_answer = faq_answers[top_idx]
            st.write(f"FAQ Answer: {faq_answer}")

            # Add an additional response from the Gemini API
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")  # Initialize the model
                response = model.generate_content(user_input)  # Generate response based on the user input
                st.write(f"LLM Response: {response.text}")  # Corrected to access 'result' attribute
            except Exception as e:
                st.write(f"Error generating response from Gemini API: {e}")

        elif intent['labels'][0] == "Product Search":
            # Placeholder for product search functionality
            # In a real application, this would query a product database or API
            st.write("Performing product search...")
            # For now, we just provide a simple response
            st.write("Showing results for products matching your query...")

        elif intent['labels'][0] == "Order Tracking":
            order_id = next((ent['word'] for ent in entities if 'order' in ent['word'].lower()), None)
            if order_id:
                st.write(f"Tracking order {order_id}...")
                # In a real application, this would query an order tracking API
                # For now, provide a simple placeholder response
                st.write(f"Order {order_id} is on the way!")
            else:
                st.write("Could not find the order ID. Please provide a valid order ID.")

        elif intent['labels'][0] == "General Chat":
            # Call Gemini API for text generation (General conversation)
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")  # Initialize the model
                response = model.generate_content(user_input)  # Generate response based on the user input
                st.write(f"Gemini Response: {response.text}")  # Corrected to access 'result' attribute
            except Exception as e:
                st.write(f"Error generating response from Gemini API: {e}")

        else:
            st.write("I'm here to help! Please ask about products, track your orders, or inquire about FAQs.")
    else:
        st.write("Please enter a question to send.")
