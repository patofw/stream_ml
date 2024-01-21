FROM python:alpine3.12
COPY . /STREAM_ML
WORKDIR /STREAM_ML

RUN python -m pip install --upgrade pip
RUN pip install --upgrade build 
RUN python -m build                               
RUN pip install .
EXPOSE 5100
CMD python ./api.py


