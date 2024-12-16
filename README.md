# SEN LEGAL ASSISTANT 
-------------------------

## Context

<p align = "justify">

This project explores the feasibility and implications of creating an artificial intelligence (AI)-based legal assistant for citizens and legal professionals in Senegal. The country, which illustrates the challenges of evolving legal systems in Africa, faces a growing demand for legal services, limited access to resources, complex procedures, and a heavy workload for legal practitioners. An AI assistant could improve judicial efficiency and make justice more accessible and understandable to the general public.
 </p>

## Methodology

<!--### RAG System

![rag_system](https://github.com/user-attachments/assets/d7519554-c2dc-42c5-82ff-fd91b1e3d739)
-->

### Agent System

![agent_system](https://github.com/user-attachments/assets/e6ffeb8e-9acb-4cae-a456-6303792a6750)


<!-- <p style="text-align: justify"> -->
<!-- In this project, we use Deep Neural Networks to identify which image is fake or real. The training will be done on a dataset that we got from Kaggle (check it here <a href="https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection?resource=download)">kaggle_real_fake_faces</a>) created by $\color{darkorange}Seonghyeon \space Nam,\space Seoung \space Wug \space Oh,\space et\space al.$ They used expert knowledge to photoshop authentic images. The fake images range between easy, medium, or hard to recognize. The modifications are made on the eyes, nose, and mouth (which permit human beings to recognize others) or the whole face. -->
<!-- </p> -->

<!-- ![fake_photoshop](https://github.com/minostauros/Real-and-Fake-Face-Detection/raw/master/filename_description.jpg) -->

<!-- The image above is described as a fake image file. The name of the file can be decomposed into three different parts separated by underscores:

- The first part indicates the quality of the Photoshop or the difficulty of recognizing that it is fake;
- The second part indicates the identification number of the image;
- The third and final part indicates the modified segment of the face in binary digits with the following signature -> $\color{orange}[left\\_eye\\_bit,\space right\\_eye\\_bit,\space nose\\_bit,\space mouth\\_bit]$. The segment is modified if it is the positive bit (1). Otherwise, the segment is not modified.  -->

### Requirements

To test our application, you must have Git and Python 3 installed on your computer if you haven't done so already. You can check if Git and Python are installed by typing the following commands in your terminal:

- For Git:
```console
git --version
```
- For Python:
```console
python3 --version
```

If they are installed, the terminal will display the corresponding version. If not, you can install Git by following the official documentation available on their site [installation of Git](https://git-scm.com/book/fr/v2/D%C3%A9marrage-rapide-Installation-de-Git). To install the latest version of Python 3, download it from the official site [installation of Python](https://www.python.org/downloads/).

After installing Git, remember to add your GitHub username and password before starting the installation step. Open your terminal and type the following commands:

- Add your name:
```
git config --global user.name "Your Name"
```
- Then add your email:
```
git config --global user.email "youremail@yourdomain.com"
```

### Installation

`SenLegalAssistant` is a platform developed with Flask to provide an intelligent legal assistant for Senegalese people. You can install it by following these steps:

- Open your terminal and navigate to the directory where you want to install the application:
```console
cd path_to_your_directory
```
- Type the following command in the terminal to clone the GitHub repository:
```console
git clone https://github.com/Oumar199/sen_legal_assistant.git
```
- Change to the cloned directory with the command:
```console
cd sen_legal_assistant
```
- Create a Python environment with `virtualenv`:
```console
pip install --user virtualenv
python3 -m venv env
```
- Add `~/.local/bin` to your `PATH` if it’s not already included (for Linux or macOS):
```console
export PATH="$HOME/.local/bin:$PATH"
```
- Activate the virtual environment (for Windows):
```console
.\env\Scripts\activate
```
- Activate the virtual environment (for Linux or macOS):
```console
source env/bin/activate
```
- Install the required libraries in your environment by typing the following command:
```console
pip install -r requirements.txt
```
- You can deactivate the environment once you are finished:
```console
deactivate
```
- To ensure the proper functioning of the models, you need to add your API keys to the .env file. The file should appear as follows after being opened with your editor:

![Capture d'écran 2024-12-16 201749](https://github.com/user-attachments/assets/74f6ae59-e6a9-4a84-9821-2a6ed0af079e)

Replace all occurrences of `# place the token in this area` and `# place the api key in this area` with the corresponding tokens and API keys from the different platforms. You can generate them from the following URLs:
- for Groq: [groq_api](https://console.groq.com/keys)
- for Mistral: [mistralai_api](https://console.mistral.ai/api-keys/)
- for Tavily: [tavily_api](https://tavily.com/)
- for Hugging Face: [hugging_face](https://huggingface.co/settings/tokens)

### Tutorial

#### Execution

Before moving on to the next step regarding the use of the platform, make sure you have Flask (see the installation step) installed in your environment. The platform will be accessible via your browser at the following address: `http://127.0.0.1:5000/`. To run the application, type the following commands:

```console
export FLASK_APP=run.py
flask run
```

Once these commands are executed, you will see the following page displayed:

![Screenshot 2024-12-07 130141](https://github.com/user-attachments/assets/77cc1762-5798-4e67-aac4-ccc26805e28c)

To stop the execution of the platform, simply type `CTRL+C` in your terminal and close it.

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
  howpublished={https://github.com/Oumar199/sen_legal_assistant},
  year={2024}
}
```
