from . import pars_args_data_frame_processor, pars_args_data_frame_filter, pars_args_data_cleaning

import sys

if __name__ == "__main__":
    pars_args_data_frame_processor(sys.argv[1:])
    pars_args_data_frame_filter(sys.argv[1:])
    # pars_args_data_cleaning(sys.argv[1:])
