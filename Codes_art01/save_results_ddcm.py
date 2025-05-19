#-----------------------------------
# General libraries
#-----------------------------------
import numpy as np
import scipy as sp
from tabulate import tabulate
import os

'''
This class has the following attributes: it computes the convergence rates
given the characteristic mesh size h and the errors. Notice that the error should be 
a dictionary whose keys are the norms and whose values are lists. Also, it generates 
a table summarizing the errors as a function of h as well as the final convergence rates per norm. 
As output, it produces a .txt file and a .npy file. The former is just for consulting, whereas 
the later is used to generate the graphs later on. 
'''

class DDCM_save_results:
    
    def __init__(self, hhs, errors, formulation='Natural'):
        self.hhs = hhs
        self.errors = errors
        self.formulation = formulation
        
    def compute_convergence_rates(self):
        '''
        Compute the convergence rate based on the two last errors and mesh sizes
        '''
        conv_rates = {}
        for key, value in self.errors.items():
            if np.all(np.isnan(value)):
                conv_rates[key] = [np.nan] 
            else:
                p = (np.log(value[-1] / value[-2])) / np.log(self.hhs[-1] / self.hhs[-2])
                conv_rates[key] = [p]  
        return conv_rates

    def save_results(self, order_poly, filename_base, foldername):
        '''
        Save the results in a table (txt formats)
        '''
        conv_rates = self.compute_convergence_rates()
        headers = ['Norm'] + [f'{size}' for size in self.hhs] + ['Convergence Rate']
        table_data = []
        
        # Define each row
        for norm, error in self.errors.items():
            row = [norm] + error + conv_rates[norm]
            table_data.append(row)
                   
        # Generate table for txt output
        table = tabulate(table_data, headers=headers, tablefmt="github")
                   
        # Print the results
        print(f"\nResults for {self.formulation} formulation ({order_poly}):")
        print(table)
        
        # We save all the results in a folder
        subfolder = os.path.join(foldername, order_poly)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
            
        # .txt file
        txt_filename = f'{filename_base}.txt'
        txt_path = os.path.join(subfolder, txt_filename)
        with open(txt_path, 'w') as f:
            f.write(table)
                
        # .npy file for further visualization
        results_dict = {
            'h': self.hhs,
            'errors': self.errors,
            'conv_rates': conv_rates
        }
        npy_filename = f'{filename_base}.npy'
        npy_path = os.path.join(subfolder,npy_filename)
        np.save(npy_path, results_dict, allow_pickle=True)
                
        return table
    
