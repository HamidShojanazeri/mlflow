from fastai.vision import *
from fastai.callbacks import mlflow
import mlflow
import mlflow.fastai


path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
model = simple_cnn((3,16,16,2))


mlflow.fastai.autolog()

learn = Learner(data, model, metrics=accuracy)

learn.fit_one_cycle(1)
