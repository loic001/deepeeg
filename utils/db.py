import uuid
import os
import logging
import shelve
import glob
import pathlib
import hashlib

class ObjectSaver(object):
    def __init__(self, db_name):
        self.db_name = db_name
        pathlib.Path(self.db_name).mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def save(self, obj, _id=None):
        _id = _id if _id else uuid.uuid4().hex
        with shelve.open(self.db_name) as db:
            db[_id] = obj
        return _id

    def load(self, _id):
        obj = None
        with shelve.open(self.db_name) as db:
            try:
                obj=db[_id]
            except: self.logger.warn('{} does not exist in db {}'.format(_id, self.db_name))
        return obj

    def remove_files(self):
        for filename in glob.glob('{}.*'.format(self.db_name)):
            os.remove(filename)

def get_or_load(data, db_load_func=None):
    if isinstance(data, str):
        if not callable(db_load_func):
            raise ValueError('db_load_func must be a defined callable')
        data_loaded = db_load_func(data)
        data = data_loaded
    return data

def get_or_load_dict(data_dict, db_load_func=None):
    data_dict = {k:get_or_load(v, db_load_func) for k, v in data_dict.items()}
    return data_dict.get('calib_dataset', None), data_dict.get('dataset', None)

def hash_string(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()
