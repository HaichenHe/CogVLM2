#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args



import generate_video_specific_prompt




def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    generate = generate_video_specific_prompt.generate_video_specific_prompt
    
    
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)
        launch_job(cfg=cfg, init_method=args.init_method, func=generate)

        

if __name__ == "__main__":
    main()
