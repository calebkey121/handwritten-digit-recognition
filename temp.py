import idx2numpy
import json
import random
from pathlib import Path

TRAIN_IMAGE_FILENAME = "./data/mnist/train-images.idx3-ubyte"
TRAIN_LABEL_FILENAME = "./data/mnist/train-labels.idx1-ubyte"

images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)

examples = []
per_digit = 5
counts = {i: 0 for i in range(10)}

indices = list(range(len(labels)))
random.shuffle(indices)

for i in indices:
    digit = int(labels[i])
    if counts[digit] >= per_digit:
        continue

    pixels = images[i].reshape(784).tolist()
    examples.append({
        "label": digit,
        "pixels": pixels
    })
    counts[digit] += 1

    if sum(counts.values()) >= per_digit * 10:
        break

out_path = Path("./mnist_examples.json")

with open(out_path, "w") as f:
    json.dump({"examples": examples}, f)

print(f"Wrote {len(examples)} MNIST examples to {out_path}")