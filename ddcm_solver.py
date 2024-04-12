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

def SetManualData():
    # Read or program some user defined data
    # ...
    return ddcm_data


def CreateSyntheticData(std_solution):

    return ddcm_data


def DDCMSolver(ddcm_data, ddcm_formulation_type):

    # Run DDCM model
    ddcm_sol = DDCM()


    # Visualize the results

    return 0


#------------------------------------------
# User defined main() function: Program your solver below

def _main_ex1_():
    
    nsd = 2
    Ls = [1.0, 1.0]
    Ns = [100, 100]
    domain = SimpleComputationalDomain(nsd, Ls, Ns)

    std_model_type = 'Darcy-Linear'
    sol_std = 



    ddcm_data = CreateSyntheticData(std_model_type)

    ddcm_model_type = 'NLSQP-5F-A'
    ir = DDCMSolver(ddcm_data, ddcm_model_type)


#------------------------------------------