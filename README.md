# NLP-Powered Customer Service Assistant

## Project Overview
This project is an NLP-powered customer service assistant designed to enhance customer interactions through intelligent query handling. The assistant integrates advanced natural language processing (NLP) techniques for intent classification, named entity recognition (NER), FAQ retrieval, and conversational capabilities using a generative language model.

## Features
1. **Intent Classification**: Uses a zero-shot classification model to detect user intents such as FAQ Inquiry, Product Search, Order Tracking, and General Chat.
2. **Named Entity Recognition (NER)**: Extracts relevant entities like order IDs from user queries.
3. **Retrieval-Augmented Generation (RAG)**: Combines a retrieval system with an FAQ dataset to provide accurate and contextually relevant answers.
4. **LLM Integration**: Utilizes a generative language model (Gemini-1.5-flash) for open-ended conversations and supplemental responses.
5. **Order Tracking API**: Implements a Flask-based API to simulate order tracking functionality.

## Familiarity with Chatbot Platforms
### Recommended Chatbot Platform: **Rasa**
#### Why Rasa?
Rasa is an open-source conversational AI platform that offers the following advantages:
- **Customizability**: Full control over the NLP pipeline and conversational flows.
- **Integration-Friendly**: Easy integration with APIs and external systems, such as the retrieval and generative models used in this project.
- **Advanced NLU Capabilities**: Supports intent recognition, entity extraction, and contextual conversation management.
- **Community Support**: Active community and extensive documentation for troubleshooting and learning.

Rasa's flexibility and emphasis on developer control make it an ideal choice for building a robust virtual assistant like this project.

## Repository Contents
1. **`rag_code.py`**: Implements the RAG pipeline for FAQ retrieval.
2. **`order_tracking.py`**: A mock Flask-based order tracking API.
3. **`app.py`**: Streamlit-based frontend integrating all features.
4. **FAQ Dataset**: Contains sample FAQ data in JSON format (`Ecommerce_FAQ_Chatbot_dataset.json`).
5. **Embeddings and FAISS Index**:
   - `faq_embeddings.npy`: Precomputed question embeddings.
   - `faq_index.index`: FAISS index for similarity search.

## Installation and Setup
### Prerequisites
- Python 3.8+
- Virtual environment (optional but recommended)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the order tracking API:
   ```bash
   python order_tracking.py
   ```
4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit app in your browser.
2. Enter a query in the text input field.
3. Click the "Send" button to process the query.
4. The app will classify the intent, extract entities, retrieve relevant FAQ answers, and optionally generate additional responses using the Gemini LLM.

## Example Interaction
### Input Query:
**"Do you offer installation services for your products?"**

### Detected Intent:
**FAQ Inquiry**

### Extracted Entities:
`[]`

### FAQ Answer:
**"Installation services are available for select products. Please check the product description or contact our customer support team for more information and to request installation services."**

### LLM Response:
**"Yes, we provide installation services for a variety of products. For further assistance, feel free to contact our support team."**

## LLM Prompts and Outputs
### Prompt:
**"Explain how AI works."**

### Generated Output:
**"Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, particularly computer systems. These processes include learning, reasoning, and self-correction. AI systems use data and algorithms to perform tasks that typically require human intelligence."**

## Future Improvements
1. Add real-time database integration for product search and order tracking.
2. Enhance the FAQ dataset for better coverage.
3. Implement user authentication and session management.
4. Explore multilingual support for global customers.

## Credits
- **NLP Models**: Hugging Face Transformers
- **Generative AI**: Google Gemini API
- **Frameworks**: Streamlit, Flask, FAISS, Rasa

## License
This project is licensed under the MIT License. See the LICENSE file for details.

