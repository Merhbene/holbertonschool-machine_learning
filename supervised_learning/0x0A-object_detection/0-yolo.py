#!/usr/bin/env python3
""" Initialize Yolo_V3 """
import tensorflow.keras as K



class Yolo:
    """ YOLO_V3"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        f = open(classes_path, "r")
        self.class_names = [c[:-1] for c in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
