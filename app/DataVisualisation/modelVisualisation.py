from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.utils import plot_model
import pickle
from app.model.modelML import data_test, labels_test
import os


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = 'modelTT.h5' if input("Do you want to analyse TT or K-Fold? (T/K): ").upper() == 'T' else 'modelKF.h5'
    model_path = os.path.join(script_dir, 'app', 'model', model_name)
    history_name = 'history_train_test_split.pkl' if model_name == 'modelTT.h5' else 'history_k_fold.pkl'
    history_path = os.path.join(script_dir, 'app', 'model', history_name)

    # Load the trained model
    model = load_model(model_path)

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    # Model Summary
    model.summary()

    # Loss and Accuracy
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Losses')

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracies')

    plt.show()

    # Confusion Matrix
    predictions = np.argmax(model.predict(data_test), axis=1)
    cm = confusion_matrix(labels_test, predictions)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    report = classification_report(labels_test, predictions, zero_division=1)
    print(report)

    # ROC and AUC
    lb = LabelBinarizer()
    lb.fit(labels_test)
    labels_test_bin = lb.transform(labels_test)
    predictions_bin = lb.transform(predictions)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(lb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(labels_test_bin[:, i], predictions_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Model Weights Visualization
    weights = model.layers[0].get_weights()[0]
    plt.figure(figsize=(10, 5))
    sns.heatmap(weights, cmap='viridis')
    plt.title('Weights of the First Layer')
    plt.show()
    weights = model.layers[1].get_weights()[0]
    plt.figure(figsize=(10, 5))
    sns.heatmap(weights, cmap='viridis')
    plt.title('Weights of the First Layer')
    plt.show()
    weights = model.layers[2].get_weights()[0]
    plt.figure(figsize=(10, 5))
    sns.heatmap(weights, cmap='viridis')
    plt.title('Weights of the First Layer')
    plt.show()
    weights = model.layers[3].get_weights()[0]
    plt.figure(figsize=(10, 5))
    sns.heatmap(weights, cmap='viridis')
    plt.title('Weights of the First Layer')
    plt.show()


    # Model Architecture Diagram
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    main()
