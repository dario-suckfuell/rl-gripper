'''
Author: Yiting CHEN
Date: 2021-10-22 01:15:14
LastEditTime: 2021-10-22 17:36:40
contact me through chenyiting716@gmail.com
'''

import os
import random
import numpy as np


class model_lib(object):
    def __init__(self, mode):
        self._model_path_list = []
        self._model_name_list = []
        self._model = {}
        if mode == 'TRAIN':
            self.dir = os.path.dirname(__file__) + '/01_TRAIN'
        elif mode == 'TRAIN_easy':
            self.dir = os.path.dirname(__file__) + '/01_TRAIN_easy'
        elif mode == 'TRAIN_hard':
            self.dir = os.path.dirname(__file__) + '/01_TRAIN_hard'
        elif mode == 'EVAL':
            self.dir = os.path.dirname(__file__) + '/02_EVAL'
        elif mode == 'EVAL_old':
            self.dir = os.path.dirname(__file__) + '/02_EVAL_old'
        elif mode == 'dummy':
            self.dir = os.path.dirname(__file__) + '/00_dummy'
        # print("loading model from {}".format(self.dir))
        self.detect_models()
        self.load_data()

    def detect_models(self):
        filelist = os.listdir(self.dir)
        for i in range(0, len(filelist)):
            file_name = filelist[i]
            
            path = os.path.join(self.dir, file_name, 'model.urdf')
            if os.path.isfile(path):
                self._model_path_list.append(os.path.realpath(path))
                self._model_name_list.append(file_name)

    def load_data(self):
        if self._model_name_list and self._model_path_list:
            for model_name, model_path in zip(self._model_name_list, self._model_path_list):
                self._model[model_name]=model_path

    @property
    def random(self):
        """
        return: return the absolute path of random model
        """
        num = len(self._model_name_list)
        random_num = np.random.randint(0, num)

        random_name = self._model_name_list[random_num]
        print("{} is selected".format(random_name))
        
        return self._model[random_name], random_name

    @property
    def model_path_list(self):
        return self._model_path_list

    @property
    def model_name_list(self):
        return self._model_name_list

    @property
    def available_models(self):
        return self._model

    def __getitem__(self, item):
        return self._model[item]



# test
"""
if __name__ == "__main__":
    model = model_lib()
    print(model.model_name_lib[10])
    print(model.random)
    print(model['pen_container_1'])
"""