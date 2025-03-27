"""
Purpose: General Tools
Author: Andy
"""


import os
import yaml
import shutil
from IPython import embed

def create_folder(path):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True) # race condition handling

    else:
        print(f"path {path} already exists.")

def load_yaml(argument):

    return yaml.load(open(argument), Loader=yaml.FullLoader)


def parse_args(all_args):

    tags = ["--", "-"]

    all_args = all_args[1:]

    if len(all_args) % 2 != 0:
        print("Argument '%s' not defined" % all_args[-1])
        exit()

    results = {}

    i = 0
    while i < len(all_args) - 1:
        arg = all_args[i].lower()
        for current_tag in tags:
            if current_tag in arg:
                arg = arg.replace(current_tag, "")
        results[arg] = all_args[i + 1]
        i += 2
    
    return results


def load_config(sys_args):

    args = parse_args(sys_args)
    
    # Load the base config
    params = load_yaml(args["config"])
    
    # Get the base path from environment variable, default to empty for container
    base_path = os.getenv('PROJECT_BASE_PATH', '')
    
    # If we're running locally (base_path is set), adjust all paths
    if base_path:
        def adjust_paths(d):
            """Recursively adjust paths in dictionary"""
            for key, value in d.items():
                if isinstance(value, dict):
                    adjust_paths(value)
                elif isinstance(value, str) and value.startswith('/develop'):
                    d[key] = os.path.join(base_path, value[1:])
        
        # Recursively adjust all paths in the config
        adjust_paths(params)
    
    # Apply command line overrides
    for key, item in args.items():
        if key in params:
            params[key] = int(item)
    
    return params
