#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on:         12th April, 2024
Last Modified on:   12th April, 2024

@authors: P. Vazao Balbao & R.F. Ausas
"""

#----------------------------------------------------------------------
# General libraries
import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
# Our libraries
from ddcm_classes import *
#----------------------------------------------------------------------


#----------------------------------------------------------------------------
# Program the different DDCM formulations
def DDCMSolver(ddcm_data, ddcm_formulation_type):

    # Run DDCM model
    if(ddcm_formulation_type == 'NLSQP-5F-A'):

        ddcm_sol = 0
    
    elif(ddcm_formulation_type == 'NLSQP-5F-B'):

        ddcm_sol = 0

    else:
        raise ValueError(ddcm_formulation_type, 'not implemented')
    

    return ddcm_sol

def StandardSolver():

    return solution


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# User defined main() functions: Program your options and problem below

def main_ex1_():
    
    # 1. Create (or import) the computational domain/mesh
    nsd = 2
    Ls = [1.0, 1.0]
    Ns = [100, 100]
    domain = ComputationalDomain(nsd, Ls, Ns, meshfile=None, flag_quads=False)
    #domain = ComputationalDomain(nsd, meshfile='cook_memb.msh', flag_quads=False)

    # The reference solution and synthetic data are both generated from a
    # a standard physical/numerical model
    
    # 2. Compute a reference solution to initially distribute the data
    ref_sol = StandardSolver(domain)
    ref_sol.solve(StdModel('Darcy-Linear', 'Dual-Mixed', ''))

    # Read or create some synthetic 
    #ddcm_data = CreateSyntheticData(std_model = 'Darcy-Linear', read_data=False)

    # 
    ddcm_sol = DDCMSolver(ddcm_data = 0)
    ddcm_sol.solve(ddcm_formulation = 'NLSQP-5F-A')

    return 0

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#--- Exectute the script
_ = main_ex1_()
