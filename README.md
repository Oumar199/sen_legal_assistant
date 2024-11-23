# DETECT FAKE FACE WITH VISION TRANSFORMER
-------------------------
<p align = "justify">

Deepfake has recently been used in the media to create false information that appears to come from verified sources and is an actual danger. It takes the source image and modifies it to simulate an existing personality's appearance. It has not any cyber security which can significantly identify them. The DeepFake is commonly generated by an encoder-decoder-based model which trains on many images of a single person. The input images to the encoder-decoder model are images of random people, and the output images must be that of the target people. Many technics guarantee that the returned image perfectly copies the visual aspect of the people behind the camera. Another kind of fake image generation only uses a Deep Neural Network generator to create nonexisting personalities. The classic Generator-Discriminator (GAN) Deep Neural Network can generate an image that can be used for "fake identity" in social networks. We cannot ideally detect them visually.
 </p>
 
 However, our project uses `Vision Transformer` and `Transfer Learning` to detect false faces like those generated on this site [ThisPersonDoesNotExist](https://this-person-does-not-exist.com/en). The model is trained on some images obtained from Kaggle and is available at the following link [KaggleFakeFace](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection?resource=download).




<span style = "background-color: white"><img src = "[https://miro.medium.com/max/828/1*qFzKC1GqOR17XaiQBex83w.webp](https://media.giphy.com/media/ATsWtUsuuFRfq8OhZ7/source.gif)"></span>

![VISION_TRANSFORMER](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/03/vit.gif)
![VISION_TRANSFORMER](https://ghost.graviti.com/content/images/size/w1000/2022/02/image-1.png)


### Context


<!-- <p style="text-align: justify"> -->
In this project, we use Deep Neural Networks to identify which image is fake or real. The training will be done on a dataset that we got from Kaggle (check it here <a href="https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection?resource=download)">kaggle_real_fake_faces</a>) created by $\color{darkorange}Seonghyeon \space Nam,\space Seoung \space Wug \space Oh,\space et\space al.$ They used expert knowledge to photoshop authentic images. The fake images range between easy, medium, or hard to recognize. The modifications are made on the eyes, nose, and mouth (which permit human beings to recognize others) or the whole face.
<!-- </p> -->

![fake_photoshop](https://github.com/minostauros/Real-and-Fake-Face-Detection/raw/master/filename_description.jpg)

The image above is described as a fake image file. The name of the file can be decomposed into three different parts separated by underscores:

- The first part indicates the quality of the Photoshop or the difficulty of recognizing that it is fake;
- The second part indicates the identification number of the image;
- The third and final part indicates the modified segment of the face in binary digits with the following signature -> $\color{orange}[left\\_eye\\_bit,\space right\\_eye\\_bit,\space nose\\_bit,\space mouth\\_bit]$. The segment is modified if it is the positive bit (1). Otherwise, the segment is not modified. 

### Installing

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
``` 

### Tutorial

A tutorial explaining how each package part was create is available in `readthedocs` and `github`. Click on the following link to access it $\longrightarrow$ [Tutorial](https://oumar199.github.io/fake_real_face_detection_docs/).

### Example of usage

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
```

### Citation
```bibtex
@misc{ok2023fakedetect,
  title={Fake and Real Face Detection},
  author={Oumar Kane},
  howpublished={https://github.com/Oumar199/fake_face_detection_ViT},
  year={2023}
}
```
