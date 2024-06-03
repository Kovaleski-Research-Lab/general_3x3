#--------------------------------------------------------------#
#  This script can either generate a dataset or reduce the 
#  dataset into volumes.
#--------------------------------------------------------------#

import os
import sys


from utils.general import load_config, parse_args
import run_datagen

def run(params):

    if params['task'] == 0:  # run data generation

        run_datagen.run(params)

    elif params['task'] == 1:
        
        from IPython import embed; embed()        
        #preprocess.run()     

    else:

        raise NotImplementedError

if __name__=="__main__":

    params = load_config(sys.argv)    

    run(params)
