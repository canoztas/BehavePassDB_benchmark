import json
import os
import random
from collections import defaultdict

# --- Configuration ---
# Path to your original, full development set JSON file
DEV_SET_PATH = '/content/drive/MyDrive/MobileB2C_Ongoing_BehavePassDB/MobileB2C_BehavePassDB_DevSet_ValSet/DevSet.json' 

# Directory where the new test set files will be saved
OUTPUT_DIR = '/content/drive/MyDrive/MobileB2C_Ongoing_BehavePassDB/MobileB2C_BehavePassDB_TestSet_Generated/'

# --- NEW ---
# Directory where the new, smaller TRAINING set will be saved
NEW_TRAIN_DIR = '/content/drive/MyDrive/MobileB2C_Ongoing_BehavePassDB/MobileB2C_BehavePassDB_DevSet_ValSet_Generated/'

# Percentage of subjects to use for the test set (e.g., 0.2 for 20%)
TEST_SET_RATIO = 0.2
# --- End of Configuration ---

def create_test_set():
    print("🚀 Starting the data splitting process...")

    # 1. Load the original dataset
    try:
        with open(DEV_SET_PATH, 'r') as f:
            full_dataset = json.load(f)
        print(f"✅ Successfully loaded dataset from {DEV_SET_PATH}")
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find the dataset at '{DEV_SET_PATH}'.")
        return

    subjects = list(full_dataset.keys())
    random.shuffle(subjects)

    # 2. Split subjects into training and testing pools
    split_index = int(len(subjects) * (1 - TEST_SET_RATIO))
    train_subjects = subjects[:split_index]
    test_subjects = subjects[split_index:]
    print(f"📊 Splitting into {len(train_subjects)} training and {len(test_subjects)} test subjects.")

    # --- MODIFIED: Save the new training set to its own directory ---
    # 3. Create and save the new, smaller training set
    new_train_set = {subject: full_dataset[subject] for subject in train_subjects}
    os.makedirs(NEW_TRAIN_DIR, exist_ok=True)
    # Get the original filename to use it in the new directory
    original_filename = os.path.basename(DEV_SET_PATH)
    new_train_path = os.path.join(NEW_TRAIN_DIR, original_filename)
    with open(new_train_path, 'w') as f:
        json.dump(new_train_set, f, indent=4)
    print(f"✅ New training set saved to: {new_train_path}")
    print("🚨 IMPORTANT: You must retrain your models using this new file!")
    # --- END OF MODIFICATION ---
    
    # (The rest of the script for creating the test set remains the same)
    
    # 4. Separate test data into enrollment and verification sets
    enrollment_data = defaultdict(dict)
    verification_data = defaultdict(dict)
    
    try:
        first_user = test_subjects[0]
        first_session_name = sorted(list(full_dataset[first_user].keys()))[0]
        tasks = list(full_dataset[first_user][first_session_name].keys())
        print(f"📝 Identified tasks: {tasks}")
    except (IndexError, KeyError) as e:
        print(f"❌ ERROR: Could not determine tasks from the dataset. Error: {e}")
        return

    for subject in test_subjects:
        all_sessions = sorted(list(full_dataset[subject].keys()))
        if len(all_sessions) < 2:
            print(f"⚠️ WARNING: Subject {subject} has fewer than 2 sessions. Skipping.")
            continue
        enrollment_session_name = all_sessions[0]
        verification_sessions = all_sessions[1:]
        enrollment_data[subject] = full_dataset[subject][enrollment_session_name]
        for session_id in verification_sessions:
            verification_data[f"{subject}_{session_id}"] = full_dataset[subject][session_id]

    # 5. Generate test files for each task
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for task in tasks:
        print(f"⚙️ Generating test files for task: '{task}'")
        comparisons = []
        labels = []
        
        valid_enroll_subjects = list(enrollment_data.keys())
        verification_keys = list(verification_data.keys())

        for enroll_subject in valid_enroll_subjects:
            genuine_verification_keys = [key for key in verification_keys if key.startswith(f"{enroll_subject}_")]
            for verif_key in genuine_verification_keys:
                comparisons.append(f"{enroll_subject} {verif_key}")
                labels.append("genuine")

            impostor_subjects = [s for s in valid_enroll_subjects if s != enroll_subject]
            num_impostor_trials_target = len(genuine_verification_keys)
            if not impostor_subjects: continue

            possible_impostor_pairs = []
            for impostor_subject in impostor_subjects:
                impostor_verif_keys = [key for key in verification_keys if key.startswith(f"{impostor_subject}_")]
                for verif_key in impostor_verif_keys:
                    possible_impostor_pairs.append(f"{enroll_subject} {verif_key}")
            
            random.shuffle(possible_impostor_pairs)
            num_to_sample = min(num_impostor_trials_target, len(possible_impostor_pairs))
            selected_impostor_pairs = possible_impostor_pairs[:num_to_sample]

            for pair in selected_impostor_pairs:
                comparisons.append(pair)
                labels.append("impostor")

        temp = list(zip(comparisons, labels))
        random.shuffle(temp)
        if not temp: continue
        comparisons, labels = zip(*temp)
        
        task_index = tasks.index(task) + 1
        
        comp_filename = os.path.join(OUTPUT_DIR, f'Comparisons_TestSet_Task{task_index}_{task}.txt')
        with open(comp_filename, 'w') as f:
            f.write('\n'.join(comparisons))
        
        label_filename = os.path.join(OUTPUT_DIR, f'task{task_index}_labels.txt')
        with open(label_filename, 'w') as f:
            f.write('\n'.join(labels))

        task_enroll_data = {s: {task: d[task]} for s, d in enrollment_data.items() if task in d}
        enroll_filename = os.path.join(OUTPUT_DIR, f'TestSet_Task{task_index}_{task}_enrolment.json')
        with open(enroll_filename, 'w') as f:
            json.dump(task_enroll_data, f)
            
        task_verif_data = {s: {task: d[task]} for s, d in verification_data.items() if task in d}
        verif_filename = os.path.join(OUTPUT_DIR, f'TestSet_Task{task_index}_{task}_verification.json')
        with open(verif_filename, 'w') as f:
            json.dump(task_verif_data, f)
            
    print(f"\n✅ All test files have been created in: '{OUTPUT_DIR}'")
    print("🎉 Process complete!")

if __name__ == '__main__':
    create_test_set()
