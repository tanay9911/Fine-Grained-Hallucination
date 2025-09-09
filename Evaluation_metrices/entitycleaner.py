import pandas as pd
import os
import ast  # to safely parse string representation of lists

# ===== Path to folder with CSVs =====
folder_path = r"C:\Users\roopa\OneDrive\Desktop\Evaluations\phi-3-mini"

# ===== List all CSV files in folder =====
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

for file_name in csv_files:
    # Skip the DrawBench file
    if "DrawBench" in file_name:
        continue

    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    if "Meta Caption_entities" not in df.columns:
        print(f" Skipping {file_name}, no 'Meta Caption_entities' column found.")
        continue

    # Process each row
    def remove_image_entity(entities_str):
        try:
            entities = ast.literal_eval(entities_str) if isinstance(entities_str, str) else entities_str
            if isinstance(entities, list):
                return [e for e in entities if e.lower() != "image"]
            return entities
        except:
            return entities_str  # if parsing fails, return original

    df["Meta Caption_entities"] = df["Meta Caption_entities"].apply(remove_image_entity)

    # Save back to same CSV
    df.to_csv(file_path, index=False)
    print(f" Processed {file_name}, removed 'image' entity where present.")
