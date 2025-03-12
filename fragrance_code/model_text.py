import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def train_model(df, save_path="models/"):
    """
    Trains a TF-IDF + KNN model using the provided dataset and saves the trained model.

    This function applies TF-IDF vectorization to transform textual data into numerical 
    feature vectors. It then trains a KNN model using cosine similarity 
    to find the most similar items. The trained model and the vectorizer are saved as 
    serialized files for future use.

    Args:
        df (pd.DataFrame): A Pandas dataframe with fragrance historical launches and a column named "Olfactive Profile" 
                           with textual descriptions.
        save_path (str, optional): The directory where the trained model and vectorizer 
                                   will be saved. Defaults to "models/".

    Returns:
        None

    """
    vectorizer = TfidfVectorizer(stop_words="english")
    feature_vectors = vectorizer.fit_transform(df["Olfactive Profile"])

    knn_model = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
    knn_model.fit(feature_vectors)

    # Save model and vectorizer
    with open(save_path + "tfidf_knn_model.pkl", "wb") as model_file:
        pickle.dump(knn_model, model_file)

    with open(save_path + "vectorizer.pkl", "wb") as vector_file:
        pickle.dump(vectorizer, vector_file)

    print("Model TF-IDF + KNN trained and saved successfully.")
