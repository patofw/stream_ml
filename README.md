# stream_ml
Stream Machine Learning for text recommendation
 
This small repository has been built to showcase a stream machine learning example, also known as online-training. 


# Installation 

1. Make sure you have the latest version of pip install: 
`python -m pip install --upgrade pip `

2. I highly recommend building a new virtual environment. You can use conda for example. stream_ml **requires** Python >= 3.12
    
    `conda create -n <NAME> python==3.12`
3. activate your virtual environment if you created it.
4. Build the module with `pip install --upgrade build` followed by `python -m build`
5. install all dependencies with `pip install .`
6. (OPTIONAL) If you want to run the `eda.ipynb` you will need to donwload the stopwords from NLTK. To to so, in a Python shell (write `python` in a terminal) download some NLTK dependencies

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

## Stream-ml 

The Stream ML library allows you to train an online-machine-learning model which gets updated with every new input. 
So for example, if a user does not interact with a post, then it learns that these type of posts are less interesting, updating the model's weights. 
This is powerful in the context of social media apps that are looking to have engaged users who are actively scrolling through an app.

**Important considerations include:**

- The model's capacity to handle millions of users seamlessly.
- The model's capability to dynamically adjust to users' evolving search patterns.
- You will receive 10 JSON files, each containing accepted and rejected sentences for a specific topic. 
- You don't have to build a word-embedding model. Use the embeddings to train a Accepted/Rejected model instead.


# Working example 

Install stream_ml as described above. Then, you can use both of the notebooks at will. 

You can also use the `Flask` app by running `python api.py`. This will forward 3 different landings: 
    - One for observing a new incoming post. You can go to your browser and type http://localhost:5100/prediction/<post number>, post number being an int between 0 and 19
    - Second one is for the prediction. If you open http://localhost:5100/prediction/<post number> you will see the prediction probability for that post. There is a simple radio button simulating the interaction of the user. You can click on the button and submit different values and see how the predictions and ROC scores are updated live.
    - The final landing is a simple page where the current ROC score is displayed: http://localhost:5100/score

# DOCKER

You can also launch this app within DOCKER and build an image. To do so you can use the `Dockerfile`` in the repo.
First install Docker if you haven't. Then, run the following: 
- `docker build -t stream_ml .` -> This will build the image. 
- `docker run --name stream_ml_app -p 5100:5100 stream_ml`
- You can now open a browser and navigate to http://localhost:5100/ following the same logic as described above.

# Code structure 
The code is divided as follows:

- **stream_ml directory**: Here, there's the core of the code developed. It is mainly composed of two Python classes. One built for exploratory data analysis and some data processing (`TextAnalytics`) and the other for an Online (stream) ML learning training system (`OnlineTraining`) which is a small PoC model which aims to solve the underlying tweet ranking challenge. 
- **Notebooks directory** Here, the EDA (`eda.ipynb`) and the Training notebooks (`model_train.ipynb`) are found. On the former, various plots and data processing pipelines are detailed. These analysis has been put into an E2E pipeline than can be executed using the `TextAnalytics` class. 
- **data directory** This is where the data given for the exercise lives.
- **models and outputs directories**: These are intended for saving the outputs and the fitted models.
- **templates**: Is where the simple html code for the `flask` app lives.







