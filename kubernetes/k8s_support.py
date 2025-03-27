import yaml
import os

from kubernetes import client, config
import subprocess


def exit_handler(params,job): # always run this script after this file ends.

    config.load_kube_config()   # python can see the kube config now. now we can run API commands.

    v1 = client.CoreV1Api()   # initializing a tool to do kube stuff.

    pod_list = v1.list_namespaced_pod(namespace = params['kube']['namespace'])    # get all pods currently running (1 pod generates a single meep sim) 

    current_group = [ele.metadata.owner_references[0].name for ele in pod_list.items if(params['kube'][job]['kill_tag'] in ele.metadata.name)]    # getting the name of the pod

    current_group = list(set(current_group))    # remove any duplicates

    for job_name in current_group:
        subprocess.run(["kubectl", "delete", "job", job_name])    # delete the kube job (a.k.a. pod)

    print("\nCleaned up any jobs that include tag : %s\n" % params['kube'][job]['kill_tag'])   

# Create: Results Folders
def create_folder(path):
    """
    Create a folder if it doesn't exist
    
    Parameters
    ----------
        path: Path to create
    """
    # Adjust path if running locally
    base_path = os.getenv('PROJECT_BASE_PATH', '')
    if base_path and path.startswith('/develop'):
        path = os.path.join(base_path, path[1:])

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"path {path} already exists.")


# Save: Template File
def save_file(path, data):
    """
    Save data to a file with path adjustment
    
    Parameters
    ----------
        path: Path to save to
        data: Data to save
    """
    # Adjust path if running locally
    base_path = os.getenv('PROJECT_BASE_PATH', '')
    if base_path and path.startswith('/develop'):
        path = os.path.join(base_path, path[1:])

    try:
        with open(path, "w") as data_file:
            data_file.write(data) 
            print(f"File saved at {path}.\n")
    except IOError as e:
        print(f"Error saving the file: {e}")
    except Exception as e:
        print(f"Unexpected error while saving file.")

# Load: Template File
def load_file(path):
    """
    Load file content with path adjustment
    
    Parameters
    ----------
        path: Path to the file to load
        
    Returns
    -------
        content: Content of the file as string
    """
    # Adjust path if running locally
    base_path = os.getenv('PROJECT_BASE_PATH', '')
    if base_path and path.startswith('/develop'):
        path = os.path.join(base_path, path[1:])

    with open(path, "r") as data_file:
        return data_file.read()

def keep_val(val):

    last_digit = val % 10
    return last_digit in [0, 1, 2]    

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
    
    params = load_yaml(args["config"])
    for key, item in args.items():
        if key in params:
            params[key] = int(item)
    
    return params 

def load_yaml(argument):

    return yaml.load(open(argument), Loader=yaml.FullLoader)

