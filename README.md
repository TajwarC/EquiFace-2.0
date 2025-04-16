## EquiFace-2.0
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
