# Machine learning pipeline deployment with FastAPI and Docker

## How to run

1. Open your terminal and navigate to the code's root folder

```
cd mlpipeline
```

2. Build the docker image


```
docker build -t pipedrive .
```

3. Run the docker image

```
docker run -p 8000:8000 pipedrive
```
<br>

4. This should now run uvicorn and you can copy below and paste it in your browser

```
http://0.0.0.0:8000
```
<br>

or alternatively,

```
http://localhost:8000
```

## API endpoints

Go to below to test out the endpoints or just glimpse the docs

```
http://localhost:8000/docs
```

The following endpoints exist:

* POST
    * `/predict`
    * `/models/load_version`
    * `/models/reload`
    * `/maintenance/retrain`

* GET
    * `/health`
    * `/maintenance/status`
    * `/maintenance/metrics`


# Postman - sample input

*You must be running the API first. Follow the `How to run` step first!*

If you want to try it out just open Postman and do a POST request to `http://localhost:8000/predict` and copy below to paste it as a body in raw JSON format.

```
{
    "Date": ["14/11/2013", "20/12/2013", "01/01/2014"],
    "Product": ["Shirt", "Hair Band", "Shirt"],
    "Gender": ["Male", "Female", "Female"],
    "Device_Type": ["Mobile", "Web", "Mobile"],
    "State": ["New York", "Washington", "New York"],
    "City": ["New York City", "Seattle", "New York City"],
    "Category": ["Clothing", "Accessories", "Clothing"],
    "Customer_Login_type": ["Member", "Guest", "Member"],
    "Delivery_Type": ["Normal Delivery", "one-day deliver", "one-day deliver"],
    "Individual_Price_US$": [10.0, 27, 10],
    "Time": ["22:44:41", "00:41:24", "15:20:33"],
    "Quantity": [500, 50, 100]
}
```

The expected output should be in this format

```
{
    "transaction_result": [],
    "probability": [],
    "model_predictions": [],
    "drift_detected": boolean,
    "drift_details": {}
}
```


# Run it in Python
*You must be running the API first. Follow the `How to run` step first!*

Instead of using Postman you can use either your terminal or the notebook `offline-inference.ipynb` which does a request in Python.

### Terminal
Navigate to the project folder

```
cd mlpipeline
```

And then run the below Python file

```
python offline_inference.py
```

### Notebook
Just go to the folder `notebooks` and run the notebook `offline-inference.ipynb`.

<br>

# Model training manual trigger

If you want to retrain manually on new data, assuming the .csv file `test_task_data.csv` is a database, you can do the following.

1. Navigate to the codebase folder in your terminal

```
cd mlpipeline
```

2. Run below with and provide a version. For instance, current versions are v1 and v2. If you provide v2 again it will create new v2 models with a timestamp in the `saved_models` folder. And then at inference time it will look for the latest models given the provided version. So you can have duplicate models which is fine!

```
python main.py --version_id=v3
```

