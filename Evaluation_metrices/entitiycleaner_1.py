import pandas as pd
import os
import ast
import numpy as np
# If this doesnt works please remove manually they are hardly 10 or something
folder = r"C:\Users\roopa\OneDrive\Desktop\Evaluations\phi-3-mini"

files_to_fix = ["meta_captions_sd_2_entities.csv", "meta_captions_sdxl_entities.csv"]

for file in files_to_fix:
    path = os.path.join(folder, file)
    df = pd.read_csv(path)

    if "Meta Caption" in df.columns and "Meta Caption_entities" in df.columns:
        fixed_entities = []
        for caption, entities in zip(df["Meta Caption"], df["Meta Caption_entities"]):
            # If caption is 0, empty, or NaN → set entities to np.nan
            if str(caption).strip() in ["0", "", "nan", "NaN"]:
                fixed_entities.append(np.nan)
            else:
                try:
                    # Convert string representation of list to actual list
                    ent_list = ast.literal_eval(entities)
                    # Remove 'image' (case-insensitive)
                    ent_list = [e for e in ent_list if e.lower() != 'image']
                except:
                    ent_list = []
                fixed_entities.append(ent_list)

        df["Meta Caption_entities"] = fixed_entities
        df.to_csv(path, index=False)
        print(f" Fixed entities in {file}")
