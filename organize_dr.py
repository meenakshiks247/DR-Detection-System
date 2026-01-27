import pandas as pd
import shutil
import os

# ==========================================
# 👇 CONFIGURATION: YOU MUST EDIT THESE 3 LINES 👇
# ==========================================

# 1. PATH TO THE FOLDER WITH THE RAW IMAGES
# (Go to your ODIR folder -> Open the images folder -> Copy the address from the top bar)
SOURCE_IMAGES_DIR = r"D:\glucoma cataract dataset\archive\preprocessed_images" 

# 2. PATH TO THE CSV FILE
# (Right-click the CSV file -> "Copy as path" -> Paste here)
CSV_PATH = r"D:\glucoma cataract dataset\archive\full_df.csv"

# 3. WHERE TO PUT THE SORTED IMAGES
# (This should be your new folder inside your project)
DESTINATION_DIR = r"E:\Major project\DR_Detection_System\dataset_generalist"

# ==========================================

def organize_images():
    # Check if files exist before starting
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"❌ ERROR: Could not find image folder at: {SOURCE_IMAGES_DIR}")
        return
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: Could not find CSV file at: {CSV_PATH}")
        return

    # Create destination folders if they don't exist
    for category in ["Cataract", "Glaucoma", "Normal", "DR"]:
        os.makedirs(os.path.join(DESTINATION_DIR, category), exist_ok=True)

    print("Loading CSV file...")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    counts = {"Cataract": 0, "Glaucoma": 0, "Normal": 0, "DR": 0}
    
    print(f"🚀 Starting organization... Moving files from {SOURCE_IMAGES_DIR}")
    
    for index, row in df.iterrows():
        # Get filenames
        left_eye = row.get('Left-Fundus')
        right_eye = row.get('Right-Fundus')
        
        target_folder = None
        
        # Decide where the image goes based on the diagnosis flags (1 indicates presence)
        try:
            if int(row.get('C', 0)) == 1:
                target_folder = "Cataract"
            elif int(row.get('G', 0)) == 1:
                target_folder = "Glaucoma"
            elif int(row.get('N', 0)) == 1:
                target_folder = "Normal"
            elif int(row.get('D', 0)) == 1:
                target_folder = "DR"
        except Exception:
            target_folder = None
            
        if target_folder:
            dest_path = os.path.join(DESTINATION_DIR, target_folder)
            
            # Move Left Eye
            if left_eye:
                src_L = os.path.join(SOURCE_IMAGES_DIR, str(left_eye))
                if os.path.exists(src_L):
                    shutil.copy(src_L, os.path.join(dest_path, os.path.basename(str(left_eye))))
                    counts[target_folder] += 1
                
            # Move Right Eye
            if right_eye:
                src_R = os.path.join(SOURCE_IMAGES_DIR, str(right_eye))
                if os.path.exists(src_R):
                    shutil.copy(src_R, os.path.join(dest_path, os.path.basename(str(right_eye))))
                    counts[target_folder] += 1
                
        if index % 100 == 0:
            print(f"Processed {index} patients...", end='\r')

    print("\n------------------------------------------------")
    print("✅ Organization Complete!")
    print(f"Images moved: {counts}")
    print("------------------------------------------------")

if __name__ == "__main__":
    organize_images()