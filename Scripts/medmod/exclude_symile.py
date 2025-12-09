import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

symile_fp = "/users/mspancho/scratch/1.0.0/data/train/"
medmod_fp = "/users/mspancho/Downloads/MedMod/mimic4extract/data/final/"

def load_symile(symile_fp):
    symile_fp = os.path.join(symile_fp, "hadm_id_train.npy")
    symile_hadm_np = np.load(symile_fp)
    print("Loaded npy file")

    symile_df = pd.DataFrame(symile_hadm_np, columns=["hadm_id"])
    print("Read Symile CSV")
    return symile_df

def load_medmod(medmod_fp):
    medmod_fp = os.path.join(medmod_fp, "all_stays.csv")
    print("Reading MedMod CSV")
    df = pd.read_csv(medmod_fp)

    print("Read MedMod CSV")
    return df[["hadm_id", "subject_id"]]

def exclude_symile(medmod_df, symile_df):
    symile_hadms = set(symile_df["hadm_id"].values)
    medmod_hadms = set(medmod_df["hadm_id"].values)
    print("Converted to sets")

    common_hadms = symile_hadms.intersection(medmod_hadms)
    print(f"Found {len(common_hadms)} common hadm_ids")
    common_subjects = medmod_df[medmod_df["hadm_id"].isin(common_hadms)]["subject_id"].values
    subject_ids = list(set(common_subjects))
    print(f"Found {len(subject_ids)} unique subject_ids to exclude")
    return common_hadms, subject_ids

def first_pass(common_hadms, subject_ids):
    # FIRST PASS: Delete common hadm_ids from dataset
    for subject_id in subject_ids:
        subject_id_pth = str(subject_id) + "/"
        subject_fp = os.path.join(medmod_fp, subject_id_pth)
        print(f"Processing subject_id: {subject_id} at path: {subject_fp}")

        try:
            with os.scandir(subject_fp) as entries:
                for entry in entries:
                    if entry.name == "diagnoses.csv":
                        df = pd.read_csv(entry.path)
                        diagnoses = df[~df["hadm_id"].isin(common_hadms)]
                        diagnoses.to_csv(entry.path, index=False)
                        print(f"Updated diagnoses for subject_id: {subject_id}")
                    elif entry.name == "events.csv":
                        df = pd.read_csv(entry.path)
                        events = df[~df["hadm_id"].isin(common_hadms)]
                        events.to_csv(entry.path, index=False)
                        print(f"Updated events for subject_id: {subject_id}")
                    elif entry.name == "stays.csv":
                        df = pd.read_csv(entry.path)
                        stays = df[~df["hadm_id"].isin(common_hadms)]
                        stays.to_csv(entry.path, index=False)
                        print(f"Updated stays for subject_id: {subject_id}")
        except FileNotFoundError:
            print(f"Error: The directory '{subject_fp}' was not found.")
            with open('missing_directories.log', 'a') as log_file:
                log_file.write(f"{subject_fp}\n")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("---------FIRST PASS COMPLETE---------")

def second_pass(subject_ids):
    # SECOND PASS: Delete empty subject directories
    for subject_id in subject_ids:
        subject_id_pth = str(subject_id) + "/"
        subject_fp = os.path.join(medmod_fp, subject_id_pth)
        print(f"Checking subject_id: {subject_id} at path: {subject_fp}")

        try:
            is_empty = True
            with os.scandir(subject_fp) as entries:
                for entry in entries:
                    if entry.name in ["diagnoses.csv", "events.csv", "stays.csv"]:
                        df = pd.read_csv(entry.path)
                        if not df.empty:
                            is_empty = False
                            break
            if is_empty:
                shutil.rmtree(subject_fp)
                print(f"Deleted empty directory for subject_id: {subject_id}")
        except FileNotFoundError:
            print(f"Error: The directory '{subject_fp}' was not found.")
            with open('missing_subjects.log', 'a') as log_file:
                log_file.write(f"{subject_fp}\n")
        except Exception as e:
            print(f"An error occurred: {e}")

        # After removing all subject directories that are empty, check if medmod_fp is empty
        if not os.listdir(medmod_fp):
            print(f"All subject directories have been removed from {medmod_fp}. Exiting cleanup.")
            break

    print("---------SECOND PASS COMPLETE---------")

def update_metadata(common_hadms, subject_ids, update_diagnosis_counts=False):
    # After second pass, need to update the metadata CSVs
    metadata_files = ["all_stays.csv", "all_diagnoses.csv", "phenotype_labels.csv", "diagnosis_counts.csv"]

    # 1. Keep track of indices of all_stays.csv that need to go (ie that contain subject_ids that no longer exist in the directory OR hadm_ids that were removed)
    stays_fp = os.path.join(medmod_fp, metadata_files[0])
    stays_df = pd.read_csv(stays_fp)
    existing_subjects = set()
    with os.scandir(medmod_fp) as entries:
        for entry in entries:
            if entry.is_dir():
                existing_subjects.add(int(entry.name))
            elif entry.name in metadata_files: 
                continue
            else:
                print(f"Warning: Unexpected file '{entry.name}' found in medmod_fp.")
                with open('unexpected_files.log', 'a') as log_file:
                    log_file.write(f"{entry.name}\n")

    # 2. Rewrite all_stays.csv without those indices
    indices_to_remove = []
    stays_df.sort_values(by='stay_id', inplace=True)
    stays_df.reset_index(drop=True, inplace=True)
    stays_size = len(stays_df)
    for row in stays_df.itertuples():
        if (row.subject_id not in existing_subjects) or (row.hadm_id in common_hadms):
            indices_to_remove.append(row.Index)
    stays_df.drop(indices_to_remove, inplace=True)
    stays_df.to_csv(stays_fp, index=False)
    stays_size -= len(stays_df)
    print(f"Updated {metadata_files[0]} by removing {stays_size} entries.")

    # 3. Do the same for all_diagnoses.csv
    diagnoses_fp = os.path.join(medmod_fp, metadata_files[1])
    diagnoses_df = pd.read_csv(diagnoses_fp)
    diagnoses_size = len(diagnoses_df)
    for row in diagnoses_df.itertuples():
        if (row.subject_id not in existing_subjects) or (row.hadm_id in common_hadms):
            diagnoses_df.drop(row.Index, inplace=True)
    diagnoses_df.to_csv(diagnoses_fp, index=False)
    diagnoses_size -= len(diagnoses_df)
    print(f"Updated {metadata_files[1]} by removing {diagnoses_size} entries.")

    # 4. Use those indices to rewrite phenotype_labels.csv
    phenotype_fp = os.path.join(medmod_fp, metadata_files[2])
    phenotype_df = pd.read_csv(phenotype_fp)
    phenotype_size = len(phenotype_df)
    for row in phenotype_df.itertuples():
        if row.Index in indices_to_remove:
            phenotype_df.drop(row.Index, inplace=True)
    phenotype_df.to_csv(phenotype_fp, index=False)
    phenotype_size -= len(phenotype_df)
    print(f"Updated {metadata_files[2]} by removing {phenotype_size} entries.")

    # 5 (optional). Update diagnosis_counts.csv
    # NOT IMPLEMENTED YET


def main():
    print("Starting script")

    symile_df = load_symile(symile_fp)
    medmod_df = load_medmod(medmod_fp)

    common_hadms, subject_ids = exclude_symile(medmod_df, symile_df)

    print("Starting first pass exclusion")
    first_pass(common_hadms, subject_ids)

    print("Starting second pass cleanup")
    second_pass(subject_ids)

    print("Updating metadata files")
    update_metadata(common_hadms, subject_ids)

    print("Script complete")

if __name__ == "__main__":    main()