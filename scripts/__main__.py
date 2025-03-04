"""
This module processes command-line arguments and invokes the appropriate function.
"""

import sys
from . import pars_args_data_frame_processor

if __name__ == "__main__":
    pars_args_data_frame_processor(sys.argv[1:])