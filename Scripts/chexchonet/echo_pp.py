import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

meta_fp = "/users/mspancho/Downloads/physionet.org/files/chexchonet/1.0.0/"

def main():
    print("Got patient df")
    fp = os.path.join(meta_fp, "metadata.csv")
    df = pd.read_csv(fp)
    print("Read CSV")
    pat_col = "patient_id"       # patient identifier
    label_col   = "composite_slvh_dlv"       # the class you want balanced (0/1)

    # patient-level dataframe: one row per patient
    pat_df = df.groupby(pat_col)[label_col].max().reset_index()
    print("Got patient df")

    X_pats = pat_df[pat_col].values
    y_pats = pat_df[label_col].values
    print("Split patient df into X and y")

    # get train and temp set
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        X_pats,
        y_pats,
        test_size=0.2,       # 80% train, 20% temp
        stratify=y_pats,
        random_state=42,
    )
    print("Got train and temp set")

    # split temp into val + test sets
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids,
        y_temp,
        test_size=0.5,       # 10% val, 10% test overall
        stratify=y_temp,
        random_state=42,
    )
    print("Got val and test set")

    # 
    train_pat_set = set(train_ids)
    val_pat_set   = set(val_ids)
    test_pat_set  = set(test_ids)

    print("Adding column to OG dataframe")
    df["split"] = "none"
    df.loc[df[pat_col].isin(train_pat_set), "split"] = "train"
    df.loc[df[pat_col].isin(val_pat_set),   "split"] = "val"
    df.loc[df[pat_col].isin(test_pat_set),  "split"] = "test"

    # quick sanity checks
    print(df["split"].value_counts())
    print(df.groupby("split")[label_col].mean())  # prevalence per split

    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = os.path.join(meta_fp, split)
        split_csv_pth = os.path.join(split_dir, f"{split}.csv")

        os.makedirs(split_dir, exist_ok=True)

        df_split = df[df["split"] == split]
        df_split.to_csv(split_csv_pth, index=False)

        missing_cxrs = []
        cxrs = df_split["cxr_filename"].tolist()
        for cxr in cxrs:
            cxr_pth = os.path.join(meta_fp, f"images/{cxr}")
            dst_pth = os.path.join(split_dir, cxr)
            if os.path.exists(dst_pth):
                print(f"Already moved file, skipping: {cxr_pth}")
                continue
            if not os.path.exists(cxr_pth):
                print(f"[WARNING] Missing file, skipping: {cxr_pth}")
                missing_cxrs.append(cxr)
                continue
            shutil.move(cxr_pth, dst_pth)

        miss_file = os.path.join(split_dir, "missing_cxrs.txt")
        with open(miss_file, "w") as f:
            for item in missing_cxrs:
                f.write(f"{item}\n")

def clean_splits():
    splits = ["train", "val", "test"]

    for split in splits:
        split_csv_pth = os.path.join(meta_fp, f"{split}/{split}.csv")
        missing_cxrs_pth = os.path.join(meta_fp, f"{split}/missing_cxrs.txt")
        
        df = pd.read_csv(split_csv_pth)

        try:
            missing_cxrs = []
            with open(missing_cxrs_pth, 'r') as missed_cxrs:
                for line in missed_cxrs:
                    missed = line.strip() # remove leading whitespace
                    if missed:
                        missing_cxrs.append(missed)

            if not missing_cxrs:
                print("Nothing to see here folks!")
                continue
            
            df_final = df[~df["cxr_filename"].isin(missing_cxrs)]
            df_final.to_csv(split_csv_pth, index=False)
        except FileNotFoundError:
            print(f"Error: The file '{missing_cxrs_pth}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    print("Processed original dataset from physionet files")
    clean_splits()
    print("Updated dataset splits to avoid parsing missing cxrs")