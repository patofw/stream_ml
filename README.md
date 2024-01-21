# stream_ml
Stream Machine Learning for text recommendation
 
This small repository has been built to showcase a stream machine learning example, also known as online-training. There are two main deliveries on this repo.

# Installation 

1. Make sure you have the latest version of pip install: 
`python -m pip install --upgrade pip `

2. I highly recommend building a new virtual environment. You can use conda for example. stream_ml **requires** Python >= 3.12
    
    `conda create -n <NAME> python==3.12`
3. activate your virtual environment if you created it.
4. install all dependencies with `pip install .`
5. In a Python shell (write `python` in a terminal) download some NLTK dependencies

```python
import nltk
nltk.download('stopwords')
```

# The use case

You are the newly appointed chief Data Science of the famous social media platform Y. The CEO, Sheldon Must has just asked you to re-work the recommendation algorithm to be able to adapt to a users input on-the-fly. 
In the Y app, you encounter millions of searches daily, necessitating a personalized user experience model. To meet this objective, the data engineering team has meticulously curated a subset of queries, each query representing a major topic.  There are 10 distinct topics and for each topic you have:

- A minimum of 20 sentences acknowledged as accepted, denoting user engagement through actions like clicks, extended reading, and sharing.
- At least 20 sentences marked as rejected, indicating instances where users clicked but immediately returned.
- A 384-dimensional embedding assigned to each sentence.

Your task involves constructing a model for **each user search**, emphasizing the elevation of accepted sentences in rankings and the demotion of rejected ones. It's crucial to bear in mind the necessity for the model's operation across millions of users and its adaptability to users' **real-time searches**. Furthermore, a database with stored embeddings is at your disposal for efficient retrieval of sentence embeddings.

**Important considerations include:**

- The model's capacity to handle millions of users seamlessly.
- The model's capability to dynamically adjust to users' evolving search patterns.
- You will receive 10 JSON files, each containing accepted and rejected sentences for a specific topic. 
- You don't have to build a word-embedding model. Use the embeddings to train a Accepted/Rejected model instead.


# Working example 

Install stream_ml as described above. Then, you can use both of the notebooks at will. 

As mentioned, this repo has been built thinking in scalabity for the future, using a modular, object oriented programming approach.

If any questions, please feel free to contact me at patofernandezw0607@gmail.com

# Code structure 
The code is divided as follows:

- **src directory**: Here, there's the core of the code developed. It is mainly composed of two Python classes. One built for exploratory data analysis and some data processing (`TextAnalytics`) and the other for an Online (stream) ML learning training system (`OnlineTraining`) which is a small PoC model which aims to solve the underlying tweet ranking challenge. 
- **Notebooks directory** Here, the EDA (`eda.ipynb`) and the Training notebooks (`model_train.ipynb`) are found. On the former, various plots and data processing pipelines are detailed. These analysis has been put into an E2E pipeline than can be executed using the `TextAnalytics` class. 
- **data directory** This is where the data given for the exercise lives.
- **models and outputs directories**: These are intended for saving the outputs and the fitted models.


It's suggested to go over the TextAnalytics.ppt before going over the code as a lot of the explanation about the choices made as well as what is aimed with the src code can be found there. 




