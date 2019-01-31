import logging

class P300DatasetTrainer():
    def __init__(self, classifier, train_data, valid_data):
        self.classifier = classifier
        self.train_data = train_data
        self.valid_data = valid_data
        self.logger = logging.getLogger(__class__.__name__)

    def run(self):
        train_data = self.train_data
        valid_data = self.valid_data
        X_train_dict = {'x': train_data.X}
        X_valid_dict = {'x': valid_data.X}
        train_result = self.classifier.fit(X_train_dict, train_data.y, X_valid_dict, valid_data.y)
        return train_result
