### Auto Logging MLFlow-Pytorch Models 

The current vesrsion of MLFLow-Pytorch auto-logging works using callbacks. A user can define the model, optimizer, loss function and data loaders as normal and import the callbacks from mlflow.pytorch.callbacks, "from mlflow.pytorch.callbacks import *" . Once they are ready, it is required to instantiate a learner object by calling get_learner function which accepts model, optimizer, loss function, train and test data loaders as follows:

•learn = get_learner(model,optimizer, loss_fn,train_loader, test_loader)

Then we instaniate an object from autolog() class as:

• run = autolog()

Now, the model can be trained by calling fit method on our run object, it accepts number of epochs and the learner object:

•run.fit(2, learn)

An exmaple of training and autologging mnist model is provided in callback_logging.py.

##### Custom call backs

In case of having a user's custom call back for logging, it is required to define the a call back class and add custom logging or even modifications on different stages of the training and pass it as cb_funcs to autolog().  

An example is shown in auto_logging_custom_callback.py







