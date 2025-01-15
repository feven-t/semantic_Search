E-commerce FAQ Chatbot with Semantic Search
Project Description

This project implements an E-commerce FAQ Chatbot that utilizes semantic search to answer customer questions based on a set of frequently asked questions (FAQ). The chatbot leverages HNSW (Hierarchical Navigable Small World) indexing with sentence embeddings from the Hugging Face Transformers library to match a user's query to the most relevant answers from a predefined FAQ dataset.

The FAQ data is stored in a JSON file, where each entry contains a question and its corresponding answer. The tool generates embeddings for the questions and stores them in an HNSW index to enable efficient and accurate semantic search based on cosine similarity.
Features:

    Load FAQ data from a JSON file
    Generate sentence embeddings for the FAQ questions
    Use an HNSW index to perform fast, accurate semantic search
    Interactive web interface with Streamlit to allow users to query the chatbot

Steps to Run the Tool
1. Install Required Libraries
   pip install streamlit hnswlib numpy torch transformers
2. Run the Streamlit Application
   streamlit run semantic3.py

Example
Query: can i return a product?

Top 3 answers:

    Question: Can I return a product if I changed my mind?

Answer: Yes, you can return a product if you changed your mind. Please ensure the product is in its original condition and packaging, and refer to our return policy for instructions.

    Question: Can I return a product if it was a final sale item?

Answer: Final sale items are usually non-returnable and non-refundable. Please review the product description or contact our customer support team to confirm the return eligibility for specific items.

    Question: Can I return a product if it was purchased as a gift?

Answer: Yes, you can return a product purchased as a gift. However, refunds will typically be issued to the original payment method used for the purchase.
   
