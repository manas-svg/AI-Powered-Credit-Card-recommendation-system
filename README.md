
---

# Credit Card Recommendation System

This project is a web-based, AI-powered credit card recommendation system that leverages machine learning and large language models (LLMs) to provide personalized credit card suggestions based on user financial profiles and preferences.

---

## Project Overview

The system uses a hybrid embedding approach combining text and numerical features of credit cards to find the best matches for users. It integrates a Streamlit frontend with a Hugging Face LLM backend to generate personalized insights and reward simulations.

---

## Features

- Personalized credit card recommendations based on user input
- Hybrid embeddings combining card descriptions and numeric features
- LLM-powered explanations for each recommended card
- Interactive Streamlit UI for easy user interaction
- Deployed on Streamlit Cloud with Hugging Face integration

---

## Project Structure

```
creditCardRecommendationSystem/
 ┣ model/
 ┃ ┣ credit_card_data_cleaned.csv
 ┃ ┣ credit_card_data_final.csv
 ┃ ┣ credit_card_embedder.joblib
 ┃ ┣ credit_card_hybrid_embeddings.npy
 ┃ ┗ credit_card_scaler.joblib
 ┣ app.py
 ┣ README.md
 ┣ requirements.txt
 ┗ sodapdf-converted.pdf
```

---


## Usage

- Enter your financial profile and preferences in the UI.
- Get personalized credit card recommendations with detailed explanations.

---

## How It Works

- **Data Layer:** Contains cleaned credit card data and precomputed embeddings.
- **Embedding Models:** Sentence Transformer model for text embeddings and scaler for numeric features.
- **Recommendation Engine:** Combines text and numeric embeddings to find best matches.
- **LLM Integration:** Uses Hugging Face Zephyr-7B-Beta model for personalized insights.
- **Frontend:** Streamlit app for user interaction.

---

## Data Processing

- Extracted credit card data from PDF using pdfplumber.
- Parsed and cleaned data with regex to handle various formats.
- Converted monetary values (lakh, crore) to numeric.
- Saved cleaned data as CSV for model training.

---

## Model Training and Embeddings

- Used 'all-MiniLM-L6-v2' sentence transformer for text embeddings.
- Normalized numeric features with MinMaxScaler.
- Combined text and numeric embeddings into hybrid vectors.
- Saved embeddings and models for inference.

---

## Recommendation Logic

- User inputs text preferences and numeric constraints.
- User input is embedded and normalized.
- Cosine similarity is computed against card embeddings.
- Top 5 cards are selected and passed to LLM for explanation.

---

## Deployment

- Deployed on Streamlit Cloud with Hugging Face integration.
- Hugging Face API token managed securely via environment variables.

---

## Future Work

- Add more cards and update dataset regularly.
- Improve LLM prompts for better explanations.
- Add user authentication and history tracking.

---


