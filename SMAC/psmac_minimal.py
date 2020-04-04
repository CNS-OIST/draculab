#! /usr/bin/env python3
# coding: utf-8

# In[1]:


# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.configspace import Configuration
from smac.scenario.scenario import Scenario
from smac.facade.psmac_facade import PSMAC
from smac.facade.smac_facade import SMAC
import numpy as np
import sys
import logging



logging.basicConfig(level=logging.DEBUG)

cs = ConfigurationSpace()
mu = UniformFloatHyperparameter("mu", 0.1, 4.0, default_value=0.5)
cs.add_hyperparameters([mu])

def eval_config(cfg):
    """ Returns the error for a network with a given configuration.
 
        Args:
            cfg : a configuration dictionary.
        Returns:
            error : A random float.
    """   
    return np.random.random() + cfg['mu']

scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 100,   # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "false",
                     "shared_model": True,
                     "ta" : eval_config,
                     "wallclock_limit" : 30.,
                     "input_psmac_dirs": "smac3_outz",
                     "output_dir" : "smac3_outz"
                     })


# In[3]:


#psmac = PSMAC(scenario=scenario, rng=np.random.RandomState(42),
#              tae=ExecuteTAFuncDict, n_optimizers=5, validate=True)
#incumbent = psmac.optimize()
#inc_value = eval_config(incumbent[0].get_dictionary())
#print("Optimized Value: %.2f" % (inc_value))


# In[4]:


#incumbent


# In[5]:


#get_ipython().run_line_magic('cd', '/home/z/projects/SMAC3')


# In[22]:


if __name__=='__main__':
    argv = sys.argv
    if len(argv) != 2:
        raise ValueError('psmac_minimal receives a single integer argument')
    try:
        seed = int(argv[1])
    except ValueError:
        print('Incorrect argument. Using default seed.')
        seed = 3

    smac = SMAC(scenario=scenario, rng=np.random.RandomState(seed),
                    tae_runner=eval_config, run_id=seed)
    smac_incumbent = smac.optimize()
    inc_value = eval_config(smac_incumbent.get_dictionary())


    #psmac = PSMAC(scenario=scenario, rng=np.random.RandomState(seed),
    #          tae=ExecuteTAFuncDict, n_optimizers=2, validate=True)
    #psmac_incumbent = psmac.optimize()
    #psmac_value = eval_config(psmac_incumbent[0].get_dictionary())

    #print("Optimized Value: SMAC=%.2f, PSMAC=%.2F" % (inc_value,psmac_value))
    print("Optimized Value: %.2f" % (inc_value))

    print(smac_incumbent.get_dictionary())
