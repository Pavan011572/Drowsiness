import os
import glob
import shutil
import random
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def reduce_image_dataset(dataset_dir, output_dir, target_count=2000, ratio=None, seed=42):
    """
    Downsamples the raw image dataset into a balanced, stratified subset.
    """
    random.seed(seed)
    categories = ['Drowsy', 'Non Drowsy']
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Reducing Image Dataset from '{dataset_dir}' ---")

    # Determine per-class target count
    if ratio is not None:
        per_class_target = None  # calculated per folder
    else:
        per_class_target = target_count // len(categories)

    total_kept = 0

    for cat in categories:
        src_folder = os.path.join(dataset_dir, cat)
        dst_folder = os.path.join(output_dir, cat)
        os.makedirs(dst_folder, exist_ok=True)

        if not os.path.exists(src_folder):
            print(f"Warning: Directory '{src_folder}' does not exist.")
            continue

        images = glob.glob(os.path.join(src_folder, "*.[pP][nN][gG]")) + \
                 glob.glob(os.path.join(src_folder, "*.[jJ][pP][gG]")) + \
                 glob.glob(os.path.join(src_folder, "*.[jJ][pP][eE][gG]"))

        if ratio is not None:
            k = int(len(images) * ratio)
        else:
            k = min(per_class_target, len(images))

        sampled_images = random.sample(images, k)
        print(f"Category '{cat}': Original = {len(images)} images -> Keeping = {len(sampled_images)} images")

        for img_path in sampled_images:
            shutil.copy2(img_path, os.path.join(dst_folder, os.path.basename(img_path)))

        total_kept += len(sampled_images)

    print(f"Reduced dataset saved to '{output_dir}' (Total images: {total_kept})")
    return output_dir


def reduce_csv_dataset(csv_path, output_csv_path, target_rows=5000, ratio=None, seed=42):
    """
    Downsamples feature CSV file with stratified sampling to preserve target ratio.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        return None

    df = pd.read_csv(csv_path, on_bad_lines='skip')
    print(f"\n--- Reducing Feature CSV '{csv_path}' ---")
    print(f"Original CSV shape: {df.shape}")

    if 'target' not in df.columns:
        print("Error: 'target' column missing in CSV.")
        return None

    if ratio is not None:
        sample_size = int(len(df) * ratio)
    else:
        sample_size = min(target_rows, len(df))

    fraction = sample_size / len(df)
    if fraction >= 1.0:
        print("Target rows count is greater than or equal to original size. Skipping reduction.")
        return df

    # Stratified downsampling based on 'target' column
    df_reduced, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df['target'],
        random_state=seed
    )

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_reduced.to_csv(output_csv_path, index=False)

    print(f"Reduced CSV shape: {df_reduced.shape}")
    print("New Class distribution:\n", df_reduced['target'].value_counts())
    print(f"Reduced CSV saved to: '{output_csv_path}'")
    return df_reduced


def main():
    parser = argparse.ArgumentParser(description="Reduce Dataset Size for Easy Transfer & Fast Training")
    parser.add_argument("--image-dir", type=str, default=r"c:\Users\surya\Downloads\drow\Driver Drowsiness Dataset (DDD)", help="Path to raw image dataset")
    parser.add_argument("--output-image-dir", type=str, default=r"c:\Users\surya\Downloads\drow\Driver Drowsiness Dataset (DDD)_reduced", help="Path to save reduced images")
    parser.add_argument("--csv-path", type=str, default=r"c:\Users\surya\Downloads\drow\data\extracted_features.csv", help="Path to extracted features CSV")
    parser.add_argument("--output-csv-path", type=str, default=r"c:\Users\surya\Downloads\drow\data\extracted_features_reduced.csv", help="Path to save reduced CSV")
    parser.add_argument("--target-images", type=int, default=2000, help="Target total image count after reduction (default: 2000)")
    parser.add_argument("--target-csv-rows", type=int, default=5000, help="Target CSV rows count after reduction (default: 5000)")
    parser.add_argument("--ratio", type=float, default=None, help="Downsample ratio (e.g. 0.1 for 10%)")
    parser.add_argument("--reduce-images", action="store_true", help="Perform image dataset reduction")
    parser.add_argument("--reduce-csv", action="store_true", help="Perform CSV dataset reduction")
    parser.add_argument("--all", action="store_true", help="Reduce both images and CSV")

    args = parser.parse_args()

    if not args.reduce_images and not args.reduce_csv and not args.all:
        args.all = True

    if args.all or args.reduce_images:
        reduce_image_dataset(args.image_dir, args.output_image_dir, target_count=args.target_images, ratio=args.ratio)

    if args.all or args.reduce_csv:
        reduce_csv_dataset(args.csv_path, args.output_csv_path, target_rows=args.target_csv_rows, ratio=args.ratio)


if __name__ == "__main__":
    main()
