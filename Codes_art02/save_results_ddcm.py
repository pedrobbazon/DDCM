#-----------------------------------
# General libraries
#-----------------------------------
import numpy as np
import scipy as sp
from tabulate import tabulate
import os

class DDCM_save_results:
    
    def __init__(self, norms , errors, formulation='VMS-Primal'):
        
        self.norms = norms
        self.errors = errors
        self.formulation = formulation
    
    def compute_convergence_rates(self):
        ''' Compute the convergence rate based on the two last errors and mesh sizes '''
        conv_rates = {key: None for key in list(self.norms)[1:]}
        
        for key, value in list(self.errors.items())[1:]:
            if np.all([val is None for val in value]):
                conv_rates[key] = np.nan
            else:
                p = (np.log(value[-1] / value[-2])) / np.log(self.errors['h'][-1] / self.errors['h'][-2])
                conv_rates[key] = p  
        return conv_rates

    def _format_table(self, conv_rates):
        headers = ['Norm'] + [f'{h}' for h in self.errors['h']] + ['Convergence Rate']
        table_data = []

        for key, value in list(self.errors.items())[1:]:
            row = [key] + value + [conv_rates[key]]
            table_data.append(row)

        table_str = tabulate(table_data, headers=headers, tablefmt="github")
        return table_str, table_data, headers
    
    def _save_txt(self, table_str, txt_path):
        """Save the formatted table as a .txt file."""
        with open(txt_path, 'w') as f:
            f.write(table_str)

    # def _save_npy(self, data_dict, npy_path):
    #     """Save the results dictionary as a .npy file."""
    #     np.save(npy_path, data_dict, allow_pickle=True)

    def save_results(self, order_poly, filename_base, foldername):
        # 1: Compute convergence rates
        conv_rates = self.compute_convergence_rates()

        # 2: Format table
        table_str, table_data, headers = self._format_table(conv_rates)

        # 3: Print to terminal
        print(f"\nResults for {self.formulation} formulation ({order_poly}):")
        print(table_str)

        # 4: Create output directory
        subfolder = os.path.join(foldername, order_poly)
        os.makedirs(subfolder, exist_ok=True)

        # 5: Save to .txt
        txt_path = os.path.join(subfolder, f'{filename_base}.txt')
        self._save_txt(table_str, txt_path)

        # 6: Save to .npy
        # npy_path = os.path.join(subfolder, f'{filename_base}.npy')
        # data_dict = {
        #     'h': self.hhs,
        #     'errors': self.errors,
        #     'conv_rates': conv_rates
        # }
        # self._save_npy(data_dict, npy_path)

        return table_str

    
    
    
