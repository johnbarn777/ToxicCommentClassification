class BaseModel:
    def train(self, train_data):
        raise NotImplementedError
    
    def evaluate(self, test_data):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
