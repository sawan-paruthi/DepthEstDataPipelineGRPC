import sys
import os
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), 'MiDaS'))

import run as midas

class DepthEstimationMiDaS:
    def __init__(self):
        pass

    def detect(self, img_path, checkpoint_path, model_type, output_path=None):
        if checkpoint_path is None:
            raise FileNotFoundError("Checkpoint path is empty")
        if checkpoint_path is not None and not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint does not exist at given path")
        if img_path is None:
            raise FileNotFoundError("Input Image path empty")
        if img_path is not None and not os.path.exists(img_path):
            raise FileNotFoundError("Image not found at given path")
        
        try:
            midas.run(input_path=img_path, output_path=output_path, model_path=checkpoint_path, model_type=model_type)

        except Exception as e:
            tb = traceback.format_exc()
            raise Exception(f"Error occurred while running MiDaS:\n{str(e)}\n\nTraceback:\n{tb}")


if __name__ == "__main__":
    midasde = DepthEstimationMiDaS()
    output = "F:\\work\\DepthEstDataPipelineGRPC\\protos"
    input = "F:\\work\\DepthEstDataPipelineGRPC\\server\\inbound"
    checkpoint = "F:\\work\\DepthEstDataPipelineGRPC\\server\\checkpoints\\midas\\dpt_beit_large_384.pt"
    midasde.detect(img_path=input, checkpoint_path=checkpoint, model_type="dpt_beit_large_384", output_path=output )