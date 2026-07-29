import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Privacy-Preserving Driver Drowsiness Detection Pipeline")
    parser.add_argument("--reduce-dataset", action="store_true", help="Downsample image dataset and CSV for fast mobile transfer")
    parser.add_argument("--extract-features", action="store_true", help="Extract MediaPipe features from dataset")
    parser.add_argument("--train-ensemble", action="store_true", help="Train Stacking Ensemble Classifier")
    parser.add_argument("--run-fl", action="store_true", help="Run Flower Federated Learning simulation")
    parser.add_argument("--run-detector", action="store_true", help="Launch OpenCV Real-Time Detection HUD")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline from data extraction to live detector")

    args = parser.parse_args()

    dataset_path = r"c:\Users\surya\Downloads\drow\Driver Drowsiness Dataset (DDD)"
    csv_features = r"c:\Users\surya\Downloads\drow\data\extracted_features.csv"
    models_dir = r"c:\Users\surya\Downloads\drow\models"

    # Default to --all if no arguments provided
    if not any(vars(args).values()):
        args.all = True

    if args.reduce_dataset:
        print("\n" + "="*60)
        print("DATASET REDUCTION (Mobile USB Transfer Optimization)")
        print("="*60)
        from src.reduce_dataset import reduce_image_dataset, reduce_csv_dataset
        reduce_image_dataset(dataset_path, dataset_path + "_reduced", target_count=2000)
        if os.path.exists(csv_features):
            reduce_csv_dataset(csv_features, csv_features, target_rows=5000)

    if args.all or args.extract_features:
        print("\n" + "="*60)
        print("STEP 1: FEATURE EXTRACTION (MediaPipe Face Mesh)")
        print("="*60)
        from src.feature_extraction import FacialFeatureExtractor
        extractor = FacialFeatureExtractor()
        extractor.process_dataset(dataset_path, csv_features)

    if args.all or args.train_ensemble:
        print("\n" + "="*60)
        print("STEP 2: STACKING ENSEMBLE MODEL TRAINING")
        print("="*60)
        from src.train_ensemble import DrowsinessEnsembleTrainer
        if os.path.exists(csv_features):
            trainer = DrowsinessEnsembleTrainer(csv_features, models_dir)
            trainer.train_and_evaluate()
        else:
            print(f"Error: {csv_features} not found. Run feature extraction first.")
            return

    if args.all or args.run_fl:
        print("\n" + "="*60)
        print("STEP 3: FEDERATED LEARNING SIMULATION (Flower Framework)")
        print("="*60)
        from src.federated_learning import run_federated_simulation
        if os.path.exists(csv_features):
            run_federated_simulation(csv_features)
        else:
            print(f"Error: {csv_features} not found. Run feature extraction first.")

    if args.all or args.run_detector:
        print("\n" + "="*60)
        print("STEP 4: OPENCV REAL-TIME DETECTION HUD")
        print("="*60)
        from src.realtime_detector import RealTimeDrowsinessDetector
        detector = RealTimeDrowsinessDetector(models_dir)
        detector.run(video_source=0)

if __name__ == "__main__":
    main()
