import os

# Path to your folder
folder = r"C:\Users\roopa\OneDrive\Desktop\Evaluations\Flux-Dev_Drawbench_images"

for filename in os.listdir(folder):
    if filename.endswith(".jpeg"):
        # Extract the number before ".jpeg"
        number = filename.split("_")[-1].replace(".jpeg", "")
        new_name = f"{number}.jpeg"

        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        # Rename
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

print(" All files renamed successfully!")
