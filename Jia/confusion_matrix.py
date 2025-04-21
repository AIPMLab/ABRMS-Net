from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import tensorflow as tf

# 设置全局默认字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'


def load_test_data(test_folder, batch_size=16):
    """
    Load test data from a specified folder.
    """
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    return test_generator


def plot_confusion_matrix(cm, class_labels):
    """
    Plot a beautiful confusion matrix with larger fonts for the annotations and labels.
    """
    sns.set(context='talk', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.2)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels,
                     annot_kws={"size": 36})  # Adjust annot_kws "size" as needed for annotation text

    # Set the tick labels font size
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=36)  # Adjust fontsize as needed for xticklabels
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=36)  # Adjust fontsize as needed for yticklabels

    plt.show()


def evaluate_model(model, test_folder):
    """
    Load a model and evaluate it on the test set.
    """
    # Load the model

    # Load test data
    test_generator = load_test_data(test_folder)

    # Get the number of samples and number of classes
    num_samples = test_generator.samples
    num_classes = test_generator.num_classes

    # Predict the whole test set
    test_generator.reset()
    predictions = model.predict(test_generator, steps=np.ceil(num_samples / test_generator.batch_size), verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Ensure the number of class labels matches the number of classes predicted by the model
    if len(class_labels) != num_classes:
        raise ValueError(
            f"Number of class labels ({len(class_labels)}) does not match number of classes predicted by the model ({num_classes}).")

    # Identify misclassified images
    misclassified_indices = np.where(predicted_classes != true_classes)[0]
    filenames = test_generator.filenames
    misclassified_filenames = [filenames[i] for i in misclassified_indices]

    # Group misclassified images by their true class
    misclassified_by_class = {}
    for i in misclassified_indices:
        true_class = class_labels[true_classes[i]]
        if true_class not in misclassified_by_class:
            misclassified_by_class[true_class] = []
        misclassified_by_class[true_class].append(filenames[i])

    # Compute confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print('Confusion Matrix:')
    print(cm)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_labels)

    # Compute classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4)
    print('Classification Report:')
    print(report)

    # Output misclassified images for each class
    print("\nMisclassified Images by Class:")
    for class_label, misclassified_files in misclassified_by_class.items():
        print(f"\nClass '{class_label}':")
        for file in misclassified_files:
            print(file)


# 使用测试集路径调用 evaluate_model 函数
test_folder = "E:\\lung cancer\\test"  # Replace with your test folder path
model = tf.keras.models.load_model('C:\\Users\\DELL\\Desktop\\jia\\bt_model')
evaluate_model(model, test_folder)
