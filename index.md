## Introduction

Inferencing refers to the reuse of a trained model to predict labels for new observations. the parameters and hyperparameters within the model will not be changed. Inference is used to request immediate or real-time predictions for individual or small numbers of data observations.

We can create real-time inferencing solutions by deploying a model as a service, hosted in a containerized platform such as Azure Kubernetes Services(AKS). 

In this article, we are going to deploy a model as a real-time inferencing service, consume a real-time inferencing service and troubleshoot service deployment.

## 1. Deloying as a real-time service

The machanism of how to deploy a model as a inferencing service in time is that, Azure uses containers as a deployment mechansism and package the model and the script code as an image, this image can be deployed to a container in the compute target, it is regardless of which type of compute target you choose: local compute, Azure cluster and so on.

Now let's first register a trained model. After successfully training a model, you should register it in your Azure machine learning workspace, it will then be able to load the model.


The model that used, it saved as diabetic_model.pkl in outputs file
```python
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the experiment run context
run = Run.get_context()

# Prepare the dataset
diabetes = pd.read_csv('./diabetes.csv')
X, y = diabetes[['Pregnancies','PlasmaGlucose','TricepsThickness']].values, diabetes['Diabetic'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model
reg = 0.1
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetic_model.pkl')

run.complete()
```

Now we have the saved model diabetic_model.pkl, we can register it using register method now:
```python
from azureml.core import Model

classification_model = Model.register(workspace=ws,
                       model_name='diabetic_model',
                       model_path='outputs/diabetic_model.pkl', # local path
                       description='A classification model')
```

Now the model diabetic_model has been registered:

![image](https://user-images.githubusercontent.com/71245576/116570103-fc581c80-a8d7-11eb-832b-e013369c5b65.png)

You actually also can use run's register_model method when you have a reference:
```python
run.register_model( model_name='classification_model',
                    model_path='outputs/diabetic_model.pkl', # run outputs path
                    description='A classification model')
```        

The model will be deployed as a service that consist of a script to load the model and return predictions for submitted data and an environment in which the script will be run. Therefore we should define the script and environment for the service.

Typically using the init function to load the model from the model registry and using the run function to generate predictions from the input data:

The init(): called when the service is initialized
The run(raw_data): called when new data is submitted to the service.
```python
import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('classification_model')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    return predictions.tolist()
```

Now you have a script to load the model and return predictions for submitted data. We need to create an environment in which to run the entry script, which we can configure using Conda or Pip configuration file. An easy way to create this file is to use a CondaDependencies class to create a default environment (which includes the azureml-defaults package and commonly-used packages like numpy and pandas), add any other required packages, and then serialize the environment to a string and save it:

```python
from azureml.core.conda_dependencies import CondaDependencies

# Add the dependencies for your model
myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

# Save the environment config as a .yml file
env_file = 'service_files/env.yml'
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)
```

Now combining the script and environment in an InferenceConfig:
```python
from azureml.core.model import InferenceConfig

classifier_inference_config = InferenceConfig(runtime= "python",
                                              source_directory = 'service_files',
                                              entry_script="score.py",
                                              conda_file="env.yml")
```              

Now we have the entry script to use the trained model and predict for new data as well as the environment. We therefore need to configure for the compute target to which the service will be deployed.

If you are deploying to an AKS cluster, you must create the cluster and a compute target for it before deploying:

```python
from azureml.core.compute import ComputeTarget, AksCompute

cluster_name = 'aks-cluster'
compute_config = AksCompute.provisioning_configuration(location='eastus')
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_completion(show_output=True)
```
With the compute target created, you can now define the deployment configuration, which sets the target-specific compute specification for the containerized deployment:
```python
from azureml.core.webservice import AksWebservice

classifier_deploy_config = AksWebservice.deploy_configuration(cpu_cores = 1,
                                                              memory_gb = 1)
```

The code to configure an ACI deployment is similar, except that you do not need to explicitly create an ACI compute target, and you must use the deploy_configuration class from the azureml.core.webservice.AciWebservice namespace. Similarly, you can use the azureml.core.webservice.LocalWebservice namespace to configure a local Docker-based service.

Now let's try deploying the model:
```python
from azureml.core.model import Model

model = ws.models['classification_model']
service = Model.deploy(workspace=ws,
                       name = 'classifier-service',
                       models = [model],
                       inference_config = classifier_inference_config,
                       deployment_config = classifier_deploy_config,
                       deployment_target = production_cluster)
service.wait_for_deployment(show_output = True)
```
For ACI or local services, you can omit the deployment_target parameter (or set it to None).


## 2. Consuming a real-time inferencing service

After deploying a real-time service you can consume it from client applications to predict labels for new data cases.

For testing, you can use the Azure Machine Learning SDK to call a web service through the run method of a WebService object that references the deployed service. Typically you send data to the run method in JSON format:

JSON file is like this:
```JSON
{
  "data":[
      [0.1,2.3,4.1,2.0], // 1st case
      [0.2,1.8,3.9,2.1],  // 2nd case,
      ...
  ]
}
```
The response from the run method is a JSON collection with a prediction for each case that was submitted in the data. The following code sample calls a service and displays the response:
```python
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Call the web service, passing the input data
response = service.run(input_data = json_data)

# Get the predictions
predictions = json.loads(response)

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i], predictions[i])
```

In production, most client applications will not include the Azure machine learning SDK and will consume the service through its REST interface. You can determine the endpoint of a deployed service in Azure machine Learning studio, or by retrieving the scoring_uri property of the Webservice object in the SDK, like this:
```python
endpoint = service.scoring_uri
print(endpoint)
```
With the endpoint known, you can use an HTTP POST request with JSON data to call the service. The following example shows how to do this using Python:
```python
import requests
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Set the content type in the request headers
request_headers = { 'Content-Type':'application/json' }

# Call the service
response = requests.post(url = endpoint,
                         data = json_data,
                         headers = request_headers)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i]), predictions[i] )
```

If you want to restrict access to your services by applying autentication. There are two kinds of authentication you can use: key and token:

![image](https://user-images.githubusercontent.com/71245576/116599147-b316c580-a8f5-11eb-9281-bc0b1edb9874.png)

By default, authentication is disabled for ACI services, and set to key-based authentication for AKS services (for which primary and secondary keys are automatically generated). You can optionally configure an AKS service to use token-based authentication (which is not supported for ACI services).

Assuming you have an authenticated session established with the workspace, you can retrieve the keys for a service by using the get_keys method of the WebService object associated with the service:

```python
primary_key, secondary_key = service.get_keys()
```
For token-based authentication, your client application needs to use service-principal authentication to verify its identity through Azure Active Directory (Azure AD) and call the get_token method of the service to retrieve a time-limited token.

To make an authenticated call to the service's REST endpoint, you must include the key or token in the request header like this:

```python
import requests
import json

# An array of new data cases
x_new = [[0.1,2.3,4.1,2.0],
         [0.2,1.8,3.9,2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Set the content type in the request headers
request_headers = { "Content-Type":"application/json",
                    "Authorization":"Bearer " + key_or_token }

# Call the service
response = requests.post(url = endpoint,
                         data = json_data,
                         headers = request_headers)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case.
for i in range(len(x_new)):
    print (x_new[i]), predictions[i] )
```

## 3. Troubleshooting service deployment

There are a lot of elements to a real-time service deployment, including the trained model, new data, environment configuration, container image and host. Troubleshooting a failed deployment or an error when consuming a deployed service can be complex.

Now, as an initial troubleshooting step you can check the status of a service by examing its state:

```python
from azureml.core.webservice import AksWebservice

# Get the deployed service
service = AksWebservice(name='classifier-service', workspace=ws)

# Check its state
print(service.state)
```

Note that to view the state of a service, you must use the compute-specific service type(for example AksWebservice) and not a generic WebService object. For an operational service, the staet should be Healthy.

Now, review service logs if the service is not healthy or you are experiencing errors when using it, you can review its logs:

```python
print(service.get_logs())
```
The logs include detailed information about the provisioning of the service, and the requests it has processed; and can often provide an insight into the cause of unexpected errors.

Deployment and runtime errors can be easier to diagnose by deploying the service as a container in a local Docker instance, like this:

```python
from azureml.core.webservice import LocalWebservice

deployment_config = LocalWebservice.deploy_configuration(port=8890)
service = Model.deploy(ws, 'test-svc', [model], inference_config, deployment_config)
```
You can then test the locally deployed service using the SDK:

```python
print(service.run(input_data = json_data))
```
You can then troubleshoot runtime issues by making changes to the scoring file that is referenced in the inference configuration, and reloading the service without redeploying it (something you can only do with a local service):

```python
service.reload()
print(service.run(input_data = json_data))
```


## Reference:

Build AI solutions with Azure Machine Learning, retrieved from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/
