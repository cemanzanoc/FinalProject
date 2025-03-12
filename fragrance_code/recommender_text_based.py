import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fragrance_code.processing_text import expand_query

def load_model(model_path="models/tfidf_knn_model.pkl", vectorizer_path="models/vectorizer.pkl"):
    """
    Loads the trained TF-IDF + KNN model and vectorizer.

    Returns:
        tuple: (knn_model, vectorizer)
    
    Raises:
        FileNotFoundError: If any required model file is missing.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        with open(model_path, "rb") as model_file:
            knn_model = pickle.load(model_file)

        with open(vectorizer_path, "rb") as vector_file:
            vectorizer = pickle.load(vector_file)

        return knn_model, vectorizer

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def recommend_by_text(query, df, top_n=10, exclude=None):
    """
    Recommends perfumes based on a text query using a TF-IDF + KNN model.

    Parameters:
    - query (str): Search keywords generated from colors, situations, and notes.
    - df (pd.DataFrame): Filtered dataset with potential fragrance matches.
    - top_n (int): Number of recommendations to return (default: 10).
    - exclude (list): List of fragrance notes to exclude from results.

    Returns:
    - pd.DataFrame: Filtered recommendations or fallback recommendations if no results are found.
    """
    try:
        # Load trained TF-IDF + KNN model
        knn_model, vectorizer = load_model()

        # **Step 1: Apply Filters Even Without a Query**
        if exclude:
            exclude = [word.lower().strip() for word in exclude]
            initial_size = df.shape[0]  # Store the dataset size before filtering
            df = df[~df["Olfactive Profile"].str.lower().apply(lambda x: any(term in x for term in exclude))]
            print(f"Excluded notes: {exclude}. Filtered {initial_size - df.shape[0]} perfumes.")

        # **If no query but filtered dataset exists, return recommendations**
        if not query.strip():
            print("No query provided. Returning top-rated perfumes after filtering.")
            if "Rating Value" in df.columns:
                return df.nlargest(top_n, "Rating Value")[["Perfume", "Brand", "Rating Value"]]
            return df.sample(top_n) if not df.empty else pd.DataFrame()

        # **Step 2: Convert Query into Vector Representation**
        query_vector = vectorizer.transform([expand_query(query)])

        # **Step 3: Find Nearest Neighbors Using KNN**
        distances, indices = knn_model.kneighbors(query_vector)

        # **Step 4: Debugging Logs**
        print(f" Query received: {query}")
        print(f"Dataset shape before indexing: {df.shape}")
        print(f"KNN returned indices: {indices}")
        print(f"KNN distances: {distances}")

        # **Step 5: Ensure Valid Indices**
        valid_indices = [i for i in indices[0] if i < len(df)]
        if not valid_indices:
            print("No valid indices returned from KNN. Applying fallback recommendations.")
            if "Rating Value" in df.columns:
                fallback_recommendations = df.nlargest(5, "Rating Value")[["Perfume", "Brand", "Rating Value"]]
                return fallback_recommendations.reset_index(drop=True) if not fallback_recommendations.empty else pd.DataFrame()
            return pd.DataFrame()

        # **Step 6: Select Recommendations Based on Valid Indices**
        recommendations = df.iloc[valid_indices].reset_index(drop=True)

        # **Step 7: Apply Final Exclusions After KNN**
        if exclude:
            recommendations = recommendations[
                ~recommendations["Olfactive Profile"].str.lower().apply(lambda x: any(term in x for term in exclude))
            ]
            print(f"Applied exclusions to KNN results: {exclude}")

        # **Step 8: Final Debugging Logs**
        print(f"Dataset size after applying exclusions: {recommendations.shape}")

        # **Step 9: Return Recommendations**
        if recommendations.empty:
            print("No results after filtering. Returning top-rated perfumes.")
            if "Rating Value" in df.columns:
                return df.nlargest(top_n, "Rating Value")[["Perfume", "Brand", "Rating Value"]]
            return df.sample(top_n) if not df.empty else pd.DataFrame()

        return recommendations.head(top_n)

    except Exception as e:
        print(f"Error in recommend_by_text: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if something fails


