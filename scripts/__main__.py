"""
This module processes command-line arguments and invokes the appropriate function.
"""

import sys
# from . import pars_args_data_frame_processor
# from . import pars_args_data_encoder
# from . import pars_args_data_scaling
from . import pars_args_data_visualization
from . import pars_args_data_outliers

if __name__ == "__main__":
    print(sys.argv[1:])
    pars_args_data_outliers(sys.argv[1:])
    # pars_args_data_visualization(sys.argv[1:])
    # pars_args_data_scaling(sys.argv[1:])
    # pars_args_data_encoder(sys.argv[1:])
    # pars_args_data_frame_processor(sys.argv[1:])