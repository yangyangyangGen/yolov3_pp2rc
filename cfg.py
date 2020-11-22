import os
from easydict import EasyDict

# _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

######
Cfg.MAX_BOX_NUM = 50
Cfg.HW_SIZE = (608, 608)
# Cfg.HW_SIZE = 608  # or (608, 320)
Cfg.BATCH_SIZE = 32


######


if __name__ == "__main__":
    pass
