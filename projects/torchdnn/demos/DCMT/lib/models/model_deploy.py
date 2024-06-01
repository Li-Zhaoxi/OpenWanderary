from lib.utils.deploy_helper import print_properties
import numpy as np

class ModelDeploy(object):
    def __init__(self, configs, models):
        super(ModelDeploy, self).__init__()
        self.template_size = configs.TRAIN.TEMPLATE_SIZE
        self.search_size = configs.TRAIN.SEARCH_SIZE
        self.stride = configs.MODEL.STRIDE
        self.score_size = round(self.search_size / self.stride)
        self.init_arch(models)
        
        self.count = 0

    def init_arch(self, model):
        self.inference = model['inference']

    def template(self, z, z_bbox):
        self.z = z
        self.z_bbox = z_bbox

    def track(self, x, savedebug = None):
        #cls, reg = self.inference[0].forward([self.z_bbox, x, self.z])
        cls, reg = self.inference[0].forward([x, self.z, self.z_bbox])
        # print('cls feature map:')
        # print_properties(cls.properties)
        # print('reg feature map:')
        # print_properties(reg.properties)
        if savedebug is not None and isinstance(savedebug, dict):
            savedebug["input_0"] = x
            savedebug["input_1"] = self.z
            savedebug["input_2"] = self.z_bbox
            savedebug["output_0"] = cls.buffer
            savedebug["output_1"] = reg.buffer

        return cls.buffer, reg.buffer
