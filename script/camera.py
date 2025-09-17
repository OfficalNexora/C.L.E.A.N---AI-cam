import cv2
import os

# Folder where you want to save pictures
save_dir = r"C:\Users\busto\Documents\AICam\dataset\images\background"
os.makedirs(save_dir, exist_ok=True)

# Function to generate the next available filename
def get_next_filename(folder, base_name="background", ext=".jpg"):
    existing_files = [f for f in os.listdir(folder) if f.startswith(base_name) and f.endswith(ext)]
    if not existing_files:
        return f"{base_name}1{ext}"
    # Extract numbers and find max
    nums = [int(f[len(base_name):-len(ext)]) for f in existing_files if f[len(base_name):-len(ext)].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return f"{base_name}{next_num}{ext}"

# Start camera
cap = cv2.VideoCapture(0)
print("Press SPACE to capture an image. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera Capture", frame)

    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC
        print("Escape hit, closing...")
        break
    elif key % 256 == 32:  # SPACE
        filename = get_next_filename(save_dir, base_name="background", ext=".jpg")
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved {filepath}")

cap.release()
cv2.destroyAllWindows()
