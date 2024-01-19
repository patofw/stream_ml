# stream_ml
Stream Machine Learning for text recommendation
 
This small repository has been built to showcase a stream machine learning example, also known as online-training. There are two main deliveries on this repo.

# Installation 

TODO 

# Working example 


In order to use the Notebooks of the Delivery simply run the setup.py (`python setup.py`) of this repo. This will install all the requirements and dependencies. Then, you can use both of the notebooks at will. 

As mentioned, this repo has been built thinking in scalabity for the future, using a modular, object oriented programming approach.

If any questions, please feel free to contact me at patofernandezw0607@gmail.com

# Code structure 
The code is divided as follows:

- **src directory**: Here, there's the core of the code developed. It is mainly composed of two Python classes. One built for exploratory data analysis and some data processing (`TextAnalytics`) and the other for an Online (stream) ML learning training system (`OnlineTraining`) which is a small PoC model which aims to solve the underlying tweet ranking challenge. 
- **Notebooks directory** Here, the EDA (`eda.ipynb`) and the Training notebooks (`model_train.ipynb`) are found. On the former, various plots and data processing pipelines are detailed. These analysis has been put into an E2E pipeline than can be executed using the `TextAnalytics` class. 
- **data directory** This is where the data given for the exercise lives.
- **models and outputs directories**: These are intended for saving the outputs and the fitted models.


It's suggested to go over the TextAnalytics.ppt before going over the code as a lot of the explanation about the choices made as well as what is aimed with the src code can be found there. 




