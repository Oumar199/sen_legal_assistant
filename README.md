# SEN LEGAL ASSISTANT 
-------------------------

## Context

<p align = "justify">

This project explores the feasibility and implications of creating an artificial intelligence (AI)-based legal assistant for citizens and legal professionals in Senegal. The country, which illustrates the challenges of evolving legal systems in Africa, faces a growing demand for legal services, limited access to resources, complex procedures, and a heavy workload for legal practitioners. An AI assistant could improve judicial efficiency and make justice more accessible and understandable to the general public.
 </p>

## Methodology

### RAG System

![rag_system](https://github.com/user-attachments/assets/d7519554-c2dc-42c5-82ff-fd91b1e3d739)

### Agent System

![agent_system](https://github.com/user-attachments/assets/e6ffeb8e-9acb-4cae-a456-6303792a6750)

### TrOCR for legal documents

![trocr_legal](https://github.com/user-attachments/assets/7760ef0d-afbc-4373-839a-bf310036480b)

<!-- <p style="text-align: justify"> -->
<!-- In this project, we use Deep Neural Networks to identify which image is fake or real. The training will be done on a dataset that we got from Kaggle (check it here <a href="https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection?resource=download)">kaggle_real_fake_faces</a>) created by $\color{darkorange}Seonghyeon \space Nam,\space Seoung \space Wug \space Oh,\space et\space al.$ They used expert knowledge to photoshop authentic images. The fake images range between easy, medium, or hard to recognize. The modifications are made on the eyes, nose, and mouth (which permit human beings to recognize others) or the whole face. -->
<!-- </p> -->

<!-- ![fake_photoshop](https://github.com/minostauros/Real-and-Fake-Face-Detection/raw/master/filename_description.jpg) -->

<!-- The image above is described as a fake image file. The name of the file can be decomposed into three different parts separated by underscores:

- The first part indicates the quality of the Photoshop or the difficulty of recognizing that it is fake;
- The second part indicates the identification number of the image;
- The third and final part indicates the modified segment of the face in binary digits with the following signature -> $\color{orange}[left\\_eye\\_bit,\space right\\_eye\\_bit,\space nose\\_bit,\space mouth\\_bit]$. The segment is modified if it is the positive bit (1). Otherwise, the segment is not modified.  -->

<!-- ### Installing

The `fake_face_detection` package contains functions and classes used for making exploration, pre-processing, visualization, training, searching for the best model, etc. It is available, and you install it with the following steps:

- Type the following command on the console to clone the GitHub repository:
```console
$ git clone https://github.com/Oumar199/fake_face_detection_ViT.git
```
- Switch to the cloned directory with the command:
```console
$ cd fake_face_detection_ViT
```
- Create a python environment with `virtualenv`:
```console
$ pip install --user virtualenv
$ python<version> -m venv env
```
- Activate the virtual environment:
```console
$ .\env\Scripts\activate
```
- Install the required libraries in your environment by typing the following command:
```console
$ pip install -r requirements.txt
```
- Install the `fake_face_detection` package with:
```console
$ pip install -e fake-face-detection
```
- You can deactivate the environment if you finish:
```console
$ deactivate
```  -->

<!-- ### Tutorial

A tutorial explaining how each package part was create is available in `readthedocs` and `github`. Click on the following link to access it $\longrightarrow$ [Tutorial](https://oumar199.github.io/fake_real_face_detection_docs/). -->

<!-- ### Example of usage

After installing the package, you can test it by creating a Python file named $\color{orange}optimization.py$ and add the following code inside the file to optimize the parameters of your objective function:
```python
# import the Bayesian optimization class
from fake_face_detection.optimization.bayesian_optimization import SimpleBayesianOptimization
import pandas as pd

"""
Create here your objective function and define your search spaces according to the Tutorial
"""

# Initialize the Bayesian optimization object
bo_search = SimpleBayesianOptimization(objective, search_spaces) # if you want to minimize the objective function set maximize = False

# Search for the best hyperparameters
bo_search.optimize(n_trials = 50, n_tests = 100)

# Print the results
results = bo_search.get_results()

pd.options.display.max_rows = 50
print(results.head(50))

```

To execute the file, write the following command in the console of your terminal:
```console
python<version> optimization.py
``` -->

### Citation
```bibtex
@misc{ok2024senlegalassistant,
  title={Sen Legal Assistant},
  author={Oumar Kane},
  howpublished={https://github.com/Oumar199/fake_face_detection_ViT},
  year={2024}
}
```
