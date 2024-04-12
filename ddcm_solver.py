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
import ddcm_classes
#----------------------------------------------------------------------


#----------------------------------------------------------------------------
# Read some experimental (Flux-Grad) data set and accommodate it 
# into the appropriate format to be consumed by ddcm_solvers
def SetManualData():
    # Read or program some user defined data
    # ...
    return ddcm_data


#----------------------------------------------------------------------------
# From a standard FEM solution of the corresponding physical problem,
# sample the (Flux-Grad) data set and accommodate it 
# into the appropriate format to be consumed by ddcm_solvers
def CreateSyntheticData(std_solution):

    return ddcm_data

#----------------------------------------------------------------------------
# Program the different DDCM formulations
def DDCMSolver(ddcm_data, ddcm_formulation_type):

    # Run DDCM model
    if(ddcm_formulation_type == 'NLSQP-5F-A'):

        ddcm_sol = 0
    
    elif(ddcm_formulation_type == 'NLSQP-5F-B'):

        ddcm_sol = 0

    else:
        raise ValueError(ddcm_formulation_type 'not implemented')
    

    return ddcm_sol


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# User defined main() function: Program your options and problem below

def _main_ex1_():
    
    nsd = 2
    Ls = [1.0, 1.0]
    Ns = [100, 100]
    domain = SimpleComputationalDomain(nsd, Ls, Ns)

    std_model_type = 'Darcy-Linear'
    sol_std = StandardSolver()

    ddcm_data = CreateSyntheticData(std_model_type)

    ddcm_model_type = 'NLSQP-5F-A'
    ddcm_sol = DDCMSolver(ddcm_data, ddcm_model_type)


#------------------------------------------