# test_labels.py
import pandas as pd

classes = pd.read_csv("data/raw/elliptic/elliptic_txs_classes.csv")
print("Class value counts:")
print(classes["class"].value_counts())

# According to Elliptic paper:
# - Fraud should be ~10% of labeled data
# - Licit should be ~90% of labeled data

total_labeled = len(classes[classes["class"] != 3])
class_1_count = len(classes[classes["class"] == 1])
class_2_count = len(classes[classes["class"] == 2])

print(f"\nLabeled nodes: {total_labeled}")
print(f"Class 1: {class_1_count} ({class_1_count / total_labeled * 100:.1f}%)")
print(f"Class 2: {class_2_count} ({class_2_count / total_labeled * 100:.1f}%)")

print("\n🔍 INTERPRETATION:")
if class_1_count < class_2_count:
    print("Class 1 (4,545) is likely ILLICIT (fraud) - smaller class")
    print("Class 2 (42,019) is likely LICIT (legitimate) - larger class")
    print("\n⚠️  Google Drive labels are OPPOSITE of Kaggle!")
else:
    print("Class 1 is likely LICIT (legitimate)")
    print("Class 2 is likely ILLICIT (fraud)")
