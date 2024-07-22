<p align="center">
    <h1 align="center">INFO-RETRIEVAL</h1>
</p>
<p align="center">
    address: https://github.com/kangchengX/info-retrieval
</p>
<p align="center">


<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
- [ Repository Structure](#-repository-structure)
- [ Data](#-data)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Tests](#-tests)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)
</details>
<hr>

##  Overview

The info-retrieval project designed a complex information retrieval system to process, analyze, and retrieve textual data efficiently. Core functionalities include data processing and normalization, visualization of term distribution patterns, and both traditional and machine learning-based retrieval techniques. By leveraging packages like nltk, Gensim, Pandas, and NumPy, as well as some designed data structures like inverted indices, the system ensures robust data manipulation, retrieval and effective feature extraction. In addition to traditional models like TD-IDF based model, BM 25 model and likelihood model, this project also introduces some simple learning-to-rank models like logistic regression, LambdaMART, and neural network models.

**Note**: In all the descriptions in the repository, some different terms express the same meaning - passages and documents (or doc); token and term

---

##  Repository Structure

```sh
└── info-retrieval/
    ├── data.py
    ├── display_tools.py
    ├── main.ipynb
    ├── readme.md
    ├── retrieve
    │   ├── learning.py
    │   └── tradition.py
    └── utils.py
```

---

## Data

The exapmle data used for this project are two .tsv files with columns (qid, pid, query, passage, relevancy).

---

##  Modules

<details open><summary>.</summary>

| File                                 | Summary |
| ---                                  | --- |
| [data.py](data.py)                   | The `data.py` module serves as the backbone for handling data operations within the information retrieval system. Its primary role is to manage loading, preprocessing, and storing data crucial for the various retrieval and analysis tasks executed by the system. 1, **Data Loading**:  data loader classes to load the data into complex dicts with collection-level and document-level statistics,  transforming raw input into structured formats suitable for processing. 2, **Data Preprocessing**: Contains utilities for cleaning and preparing data. 3, **Integration Hooks**: Offers integrations with other components of the repository by creating classes of views of the complex dicts which behave like simple dicts, ensuring smooth data flow and consistency across different modules involved in the information retrieval pipeline. |
| [display_tools.py](display_tools.py) | Visualize_frequency_zipfian function generates visual comparisons between normalized frequency data and Zipfian distributions in both linear and logarithmic scales, aiding in the evaluation of term distribution patterns.|
| [main.ipynb](main.ipynb)             | This notebook show some simple work flows of information retreival for this project. |
| [utils.py](utils.py)                 | Provide utility functions and a lemmatizer class with cached moethod for text processing. Implement normalized frequency calculation, token generation, and evaluation metrics like average precision and NDCG for information retrieval tasks, enhancing the overall functionality and robustness of the project.|

</details>

<details open><summary>retrieve</summary>

| File                                  | Summary |
| ---                                   | --- |
| [tradition.py](retrieve\tradition.py) | 1, **Scorer Class**: The core component, responsible for calculating retrieval scores. This class implements various scoring functions, including TF-IDF, BM 25 score, and log likelihood with smoothing methods, to evaluate the relevance of document-query pairs by utilizing the `DataLoader` from the `data.py` module to access essential data structures. 2, **Traditional Retreival class**: Achieve retrieval based on the score class. 3, **Score fuction**: score functions of these traditional retrieval models which can be used more generally. Overall, this file plays a critical role in the repository by enabling the traditional information retrieval methodologies.| |
| [learning.py](retrieve\learning.py)   | Facilitates machine learning based information retrieval by defining models like Logistic Regression, LambdaMART, and MLP, along with a Trainer to manage training processes. Integrates with the parent repositorys architecture to enhance retrieval performance using advanced model-based techniques for predicting and ranking relevancy. |

</details>

---

##  Getting Started

### System Requirements

See `requirements.txt`.

###  Data

The training data must have columns (qid, pid, query, passage). The validation data must have columns (qid, pid, query, passage, relevancy)


###  Usage

See examples in `main.ipynb`
