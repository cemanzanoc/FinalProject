import pandas as pd
import random
from fragrance_code.data_loader import load_data
from fragrance_code.recommender_text_based import recommend_by_text

# **Step 1: Define Color & Situation Mappings**
COLOR_TO_KEYWORDS = {
    "red": ["spicy", "oriental", "woody", "ambery"],
    "orange": ["citrus", "fresh", "fruity", "floral"],
    "yellow": ["floral", "white floral", "powdery"],
    "green": ["herbal", "green", "earthy", "aromatic"],
    "blue": ["aquatic", "marine", "fresh", "aromatic"],
    "purple": ["gourmand", "sweet", "balsamic", "floral"],
    "pink": ["floral", "fruity", "white floral"],
    "black": ["woody", "leathery", "musky", "oriental"],
    "white": ["white floral", "aldehydic", "powdery", "musk"],
    "beige": ["powdery", "balsamic", "vanilla", "gourmand"],
    "brown": ["woody", "spicy", "gourmand", "tobacco"],
    "gray": ["metallic", "aldehydic", "others"],
    "cyan": ["fresh", "aquatic", "marine"],
    "magenta": ["floral", "fruity", "sweet"],
    "dark red": ["spicy", "oriental", "woody", "ambery", "leather"]
}

SITUATION_TO_ACCORD = {
    "night": ["woody", "spicy", "leather", "floral", "white floral", "gourmand", "tobacco", "ambery"],
    "casual": ["fresh", "citrus", "aquatic", "fruity", "floral", "green", "aromatic"],
    "romantic": ["floral", "sweet", "musky", "woody", "spicy", "gourmand", "ambery", "aromatic"],
    "sport": ["green", "aquatic", "ozonic", "citrus"],
    "office": ["clean", "powdery", "aldehyde", "floral", "citrus", "fruity", "green", "marine", "gourmand"]
}

def recommend_fragrances(
    tagged_colors=None, situation=None, gender=None, brand=None, favorite_notes=None, exclude_notes=None, dataset_path="data/fragrance_ML_model.csv", num_recommendations=5):
    """
    Recommends fragrances based on detected colors, situation, gender, brand, and preferred fragrance notes.
    If no image is uploaded (tagged_colors is None or empty), the function uses only text-based filtering.

    Parameters:
    - tagged_colors (list): Colors detected from the uploaded image.
    - situation (str): Context for using the fragrance (e.g., "casual", "office").
    - gender (str): Preferred gender category for the fragrance ("male", "female", "unisex").
    - brand (str): Preferred brand name.
    - favorite_notes (list): A list of preferred fragrance notes.
    - exclude_notes (list): A list of fragrance notes to exclude.
    - dataset_path (str): Path to the fragrance dataset CSV file.
    - num_recommendations (int): Number of recommendations to return.

    Returns:
    - list: A randomized list of recommended fragrances.
    """

    # **Step 1: Load the fragrance dataset**
    df = load_data(dataset_path)

    # Ensure the dataset is successfully loaded
    if df is None or df.empty:
        print("Error: Fragrance database could not be loaded.")
        return ["Error: Fragrance database could not be loaded."]

    # Verify required columns exist in the dataset
    required_columns = ["Perfume", "Brand", "Gender", "Olfactive Profile", "Family"]
    if any(col not in df.columns for col in required_columns):
        print(f"The dataset is missing required columns: {required_columns}")
        return [f"Error: The dataset is missing required columns: {required_columns}"]

    # **Step 2: Extract Keywords from Colors, Situation, and Notes**
    keywords = set()

    # Extract fragrance accords based on detected colors
    if tagged_colors:
        print(f"Colors to match with fragrance characteristics: {tagged_colors}")
        keywords.update(sum([COLOR_TO_KEYWORDS.get(color, []) for color in tagged_colors], []))

    # Add situation accords
    if situation:
        print(f"Applying Situation Filter: {situation}")
        keywords.update(SITUATION_TO_ACCORD.get(situation, []))

    # Add favorite fragrance notes
    if favorite_notes:
        print(f"Adding Favorite Notes: {favorite_notes}")
        keywords.update(map(str.lower, map(str.strip, favorite_notes)))

    # **Step 3: Apply Filters Even Without a Query**

    # **Filter by Gender**
    if gender:
        gender_map = {"female": "women", "male": "men", "feminine": "women", "masculine": "men"}
        mapped_gender = gender_map.get(gender.strip().lower(), gender.strip().lower())

        df["Gender"] = df["Gender"].str.strip().str.lower()
        df = df[df["Gender"] == mapped_gender]

        if df.empty:
            print(f"No exact matches for gender '{mapped_gender}'. Expanding to unisex perfumes.")
            df = df[df["Gender"] == "unisex"]

    print(f"Dataset size after gender filter: {df.shape}")

    # **Filter by Brand**
    if brand:
        exact_match_df = df[df["Brand"].str.lower().fillna("") == brand.lower()]
        
        if exact_match_df.empty:
            print(f"No exact match for brand '{brand}'. Expanding search to partial matches.")
            df = df[df["Brand"].str.contains(brand, case=False, na=False)]
        else:
            df = exact_match_df

    print(f"Dataset size after brand filter: {df.shape}")

    # **Step 4: Apply Exclusions**
    if exclude_notes:
        exclude_notes = [note.lower().strip() for note in exclude_notes]
        initial_size = df.shape[0]  # Store dataset size before exclusion filtering
        df = df[~df["Olfactive Profile"].str.lower().apply(lambda x: any(term in x for term in exclude_notes))]
        print(f"Excluded notes: {exclude_notes}. Removed {initial_size - df.shape[0]} perfumes.")

    # **If no filters applied, return top-rated perfumes instead of random selection**
    if df.empty:
        print("No results after filtering. Returning top-rated perfumes.")
        if "Rating Value" in df.columns:
            return df.nlargest(num_recommendations, "Rating Value")["Perfume"].tolist()
        return ["No results found."]

    # **Step 5: Use Text-Based Recommender**
    query = " ".join(keywords)
    print(f"Query for Text-Based Search: {query}")

    recommended_fragrances = recommend_by_text(query, df, top_n=max(10, num_recommendations * 2), exclude=exclude_notes)

    # **Fallback if no recommendations are found**
    if recommended_fragrances.empty:
        print("No results from text-based matching. Falling back to top-rated perfumes.")
        if "Rating Value" in df.columns:
            return df.nlargest(num_recommendations, "Rating Value")["Perfume"].tolist()
        return ["No results found."]

    # **Step 6: Shuffle & Return Recommendations**
    recommended_list = list(dict.fromkeys(recommended_fragrances["Perfume"].tolist()))
    random.shuffle(recommended_list)

    return random.sample(recommended_list, min(num_recommendations, len(recommended_list)))

