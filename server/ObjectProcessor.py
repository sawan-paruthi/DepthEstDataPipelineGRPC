import os
import sys
import torch
from detectors.DepthEstimationMiDaS import DepthEstimationMiDaS
import logging


class ObjectProcessor:
    def __init__(self, prefix):
        self.model = None  
        self.detector = None
        self.prefix = prefix

    def load_model(self, model):
        # print(f"model name in object processor is {model}")
        self.detector = model

    def detect_objects(self, image_path):
        if self.detector[:5]==f"{self.prefix}":
        
            try:
                self.detector = self.detector[6:]
                logging.info(f"ObjectProcessor: Model Loaded: {self.detector}")
                abs_path = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(abs_path, "checkpoints", "midas", self.detector)
                img_path = os.path.join(abs_path, "inbound")
                print(f"model type: {self.detector[:-3]}")
                Model = DepthEstimationMiDaS()
                detections = Model.detect(img_path=img_path, checkpoint_path=model_path, model_type=self.detector[:-3])
                return detections
            
            except Exception as e:
                logging.error("ObjectProcessor: Error while executing MiDaS: {e}")
                raise Exception(f"Error while executing MiDaS:")

        else:
            logging.error("ObjectProcessor: Model not available")
            raise ValueError(f"Model not available : {self.detector}")

