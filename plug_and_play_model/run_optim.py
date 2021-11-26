# -*- coding: utf-8 -*-
"""

EHDO - ENERGY HUB DESIGN OPTIMIZATION Tool

Developed by:   E.ON Energy Research Center, 
                Institute for Energy Efficient Buildings and Indoor Climate, 
                RWTH Aachen University, 
                Germany
               
Contact:        Marco Wirtz 
                marco.wirtz@eonerc.rwth-aachen.de

"""


import load_params
import optim_model
#import post_processing
import os

          

# Load parameters
param, devs, dem, result_dict = load_params.load_params()

    
# Run optimization
result_dict = optim_model.run_optim(devs, param, dem, result_dict)

    
# Run post-processing
#post_processing.run(dir_results)
#post_processing.run(os.path.join(os.path.abspath(os.getcwd()), "Results", "test"))

    