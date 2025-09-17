import os
import cv2
from PIL import Image
import shutil

# -----------------------
folders = ["dataset/images/paper", "dataset/images/plastic"]
IMG_SIZE = (224, 224)
MARGIN = 10  # extra padding around the object
BACKUP_FOLDER = "dataset/originals"  # folder to move originals

# Make sure backup folder exists
os.makedirs(BACKUP_FOLDER, exist_ok=True)

def crop_largest_object(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv_img
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x = max(x - MARGIN, 0)
    y = max(y - MARGIN, 0)
    w = min(w + 2*MARGIN, cv_img.shape[1] - x)
    h = min(h + 2*MARGIN, cv_img.shape[0] - y)
    return cv_img[y:y+h, x:x+w]

# -----------------------
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg"))]
    label = os.path.basename(folder)

    # Detect existing numbers to avoid overwriting
    existing_numbers = set()
    for f in files:
        if f.startswith(label) and f.endswith(".png"):
            try:
                num = int(f.replace(label, "").replace(".png",""))
                existing_numbers.add(num)
            except:
                pass

    counter = 1
    for file in files:
        old_path = os.path.join(folder, file)

        # Skip if file already matches the new naming pattern
        if file.startswith(label) and file.endswith(".png"):
            continue

        # Find next available number
        while counter in existing_numbers:
            counter += 1

        new_name = f"{label}{counter}.png"
        new_path = os.path.join(folder, new_name)

        try:
            cv_img = cv2.imread(old_path)
            if cv_img is None:
                print(f"Failed to read {file}")
                continue

            cropped = crop_largest_object(cv_img)
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            pil_img = pil_img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            pil_img.save(new_path, "PNG")
            print(f"{file} -> {new_name}")

            # Move original to backup folder safely
            backup_path = os.path.join(BACKUP_FOLDER, file)
            if not os.path.exists(backup_path):
                shutil.move(old_path, backup_path)
                print(f"Original {file} moved to {BACKUP_FOLDER}")
            else:
                print(f"Original {file} already in backup, skipping move")

        except Exception as e:
            print(f"Failed to process {file}: {e}")

        existing_numbers.add(counter)
        counter += 1

print("All images safely processed, renamed, cropped, and originals backed up!")
# -----------------------   
# Note: You can now delete the BACKUP_FOLDER if everything looks good.