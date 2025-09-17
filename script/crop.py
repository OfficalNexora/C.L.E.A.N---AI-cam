import os

# Paths to your folders
folders = ["dataset/images/paper", "dataset/images/plastic"]

for folder in folders:
    files = os.listdir(folder)
    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    cropped_files = set(f for f in files if "crop" in f.lower())
    
    for f in files:
        # If it's not cropped but has a cropped version, delete it
        if f not in cropped_files:
            name, ext = os.path.splitext(f)
            # Check if a cropped version exists
            possible_crop = f"{name}_crop{ext}"
            if possible_crop in files:
                os.remove(os.path.join(folder, f))
                print(f"Deleted non-cropped duplicate: {f}")
