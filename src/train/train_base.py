class BaseTrainer:
    def __init__(self) -> None:
        pass

    def train(self, *args):
        raise NotImplementedError()

    def predict(self, *args):
        raise NotImplementedError()

    def preprocess_dataset(self):
        raise NotImplementedError()

    def save(self, *args, **kwargs):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()
