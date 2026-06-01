# EquiFace-2.0

<a href="url"><img src="https://github.com/TajwarC/VeriFair/blob/main/logo.png" align="centre" height="160" width="160" ></a>

EquiFace is a fairness benchmarking tool for biometric models used in facial verification. It requires just two inputs, the model in a ```.tflite``` format, and a testing dataset in the following format:

```
.
└── Testing Dataset/
    ├── Group A/
    │   ├── ID_1/
    │   │   ├── img_1
    │   │   ├── img_2
    │   │   ├── ...
    │   │   └── img_k
    │   ├── ID_2/
    │   │   ├── img_1
    │   │   ├── img_2
    │   │   ├── ...
    │   │   └── img_k
    │   ├── ...
    │   └── ID_n
    ├── Group B/
    │   ├── ID_1/
    │   │   ├── img_1
    │   │   ├── img_2
    │   │   ├── ...
    │   │   └── img_k
    │   ├── ID_2/
    │   │   ├── img_1
    │   │   ├── img_2
    │   │   ├── ...
    │   │   └── img_k
    │   ├── ...
    │   └── ID_n
    ├── ...
    └── Group Z
```
Where there are A-Z groups (such as skin tone or ethnic groups), each containing n individuals, and k images per individual.

The False Negative Rate (FNR) is calculated by taking an input pair from each ID (e.g. img_1 and img_2), then computing the cosine similarity between the embeddings. A false positive occurs when an input pair is not verified for a particular ID. This is done for all input pairs, for all IDs in each group.

The False Positive Rate (FPR) is calculated similarily, except input pairs are now images of one ID with another.
## Installation

### Using `uv` (Recommended)

To set up the development environment with all dependencies:

```bash
uv sync
```

To run the package as a CLI:

```bash
uv run equiface
```

### Using Jupyter Notebooks

To use EquiFace within a Jupyter notebook, you can install the package in your current environment:

```bash
uv pip install .
```

Or, if you are using `uv` to manage your environment, you can run Jupyter through `uv`:

```bash
uv run jupyter notebook
```
Calculating FPRs and FNRs
```python
# Imports
from equiface.verification import FPR, FNR

# Directories and parameters
# Set dataset_dir to None to use the default ControlFace10k dataset from Hugging Face
dataset_dir = 'testing_dataset/group_1' 
model_path = 'model.tflite'
image_size = (160,160) # Input dimension for model
threshold = 0.5 # Threshold for cosine similarity

# FNR
FNR(dataset_dir,
    model_path,
    image_size=image_size,
    threshold=threshold,
    percentage=100,
    use_multiprocessing=True,
    num_cores=4)

FPR(dataset_dir,
    model_path,
    image_size=image_size,
    threshold=threshold,
    percentage=100,
    use_multiprocessing=True,
    num_cores=4)
```

The results are saved into a YAML file:

```
- False Negatives: 67
  dataset: testing_dataset/group_1
  metric: FNR
  model_name: model
  num_selected: 1251
  total_pairs: 2502
  value: 0.0536
- False Positives: 699
  dataset: testing_dataset/group_1
  metric: FPR
  model_name: model
  num_selected: 1563
  total_pairs: 3126249
  value: 0.4472

```
We also provide templates for converting TensorFlow models into the TFLite format, using model architecture and weights provided by DeepFace [1].

[1] S. Serengil and A. Ozpinar, "A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules", Journal of Information Technologies, vol. 17, no. 2, pp. 95-107, 2024.
