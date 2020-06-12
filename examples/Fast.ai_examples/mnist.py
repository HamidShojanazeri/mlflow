from fastai.vision import *
from fastai.callbacks import mlflow
import mlflow
import mlflow.fastai


path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
model = simple_cnn((3,16,16,2))


class MLFlowTracking(LearnerCallback):
    "A `LearnerCallback` that tracks the loss and other metrics into MLFlow"
    def __init__(self, learn:Learner):
        super().__init__(learn)
        self.learn = learn
        self.metrics_names = ['train_loss', 'valid_loss'] + [o.__name__ for o in learn.metrics]

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Send loss and other metrics values to MLFlow after each epoch"
        if kwargs['smooth_loss'] is None or kwargs["last_metrics"] is None:
            return
        metrics = [kwargs['smooth_loss']] + kwargs["last_metrics"]
        for name, val in zip(self.metrics_names, metrics):
            mlflow.log_metric(name, np.float(val))
    def on_train_end(self, **kwargs: Any) -> None:
        mlflow.fastai.log_model(self.learn, "models")


learn = Learner(data, model, metrics=accuracy,
                   callback_fns=[partial(MLFlowTracking)])
with mlflow.start_run():

    learn.fit_one_cycle(1)
