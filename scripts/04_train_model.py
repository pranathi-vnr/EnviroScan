import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import config
import warnings
import logging

warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_model_training():
    logging.info("Starting model training")
    
    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        test_df = pd.read_csv(config.TEST_FILE)
        logging.info("Training and testing data loaded")
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        return

    available_features = [col for col in config.FEATURE_COLS if col in train_df.columns]
    X_train = train_df[available_features]
    y_train_raw = train_df[config.TARGET_COL]
    X_test = test_df[available_features]
    y_test_raw = test_df[config.TARGET_COL]
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    num_classes = len(le.classes_)
    logging.info(f"Training models for {num_classes} pollution sources: {list(le.classes_)}")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=config.RANDOM_STATE, class_weight='balanced')
    }

    best_model_name = None
    best_accuracy = 0.0
    best_model = None

    logging.info("Training models...")
    for name, model in models.items():
        logging.info(f"Training {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"{name} accuracy: {accuracy:.2%}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    logging.info(f"Best model: {best_model_name} with {best_accuracy:.2%} accuracy")

    logging.info("Generating performance report")
    y_pred_best = best_model.predict(X_test)
    report_str = classification_report(
        y_test, y_pred_best, target_names=le.classes_, labels=range(num_classes), zero_division=0
    )
    
    with open(config.EVALUATION_FILE, "w") as f:
        f.write(f"{best_model_name} Model Evaluation Report\n")
        f.write("="*30 + "\n")
        f.write(report_str)

    cm = confusion_matrix(y_test, y_pred_best, labels=range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix for {best_model_name}')
    plt.savefig(config.CONFUSION_MATRIX_FILE)
    plt.close()

    feature_importances = pd.Series(best_model.feature_importances_, index=available_features).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title(f'Feature Importance ({best_model_name})')
    plt.savefig(config.FEATURE_IMPORTANCE_FILE)
    plt.close()
    
    logging.info("Performance report and charts saved")

    logging.info(f"Saving winning model: {best_model_name}")
    joblib.dump(best_model, config.MODEL_FILE)
    joblib.dump(le, config.ENCODER_FILE)
    logging.info("Model saved successfully")
    
    logging.info("Model training completed")

if __name__ == "__main__":
    run_model_training()