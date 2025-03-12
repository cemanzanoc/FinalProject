import pandas as pd

import pandas as pd

def load_data(file_path="data/fragrance_ML_model.csv"):
    """
    Loads the fragrance dataset, ensures correct column names, and handles potential errors.
    
    Returns:
        pd.DataFrame: The processed dataset, or None if loading fails.
    """
    try:
        #Load the dataset (try comma separator, fallback to semicolon)
        df = pd.read_csv(file_path, sep=",") if pd.read_csv(file_path, nrows=1).shape[1] > 1 else pd.read_csv(file_path, sep=";")

        #Fix column names (remove spaces, convert to lowercase)
        df.columns = df.columns.str.strip().str.lower()

        #Rename columns to expected names
        rename_dict = {
            "perfume": "Perfume",
            "brand": "Brand",
            "gender": "Gender",
            "family": "Family",
            "olfactive profile": "Olfactive Profile"
        }
        df.rename(columns=rename_dict, inplace=True)

        #Check for missing required columns
        required_columns = ["Perfume", "Brand", "Gender", "Olfactive Profile", "Family"]
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Missing columns -> {[col for col in required_columns if col not in df.columns]}")
            return None

        print("Dataset loaded successfully!")
        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

