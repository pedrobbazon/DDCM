# ===============================================
# General libraries
# ===============================================
import numpy as np
from tabulate import tabulate
import os

class DDCMSaveResults:
    def __init__(self, errors: dict, formulation: str):
        self.errors = errors
        self.formulation = formulation

    def compute_convergence_rates(self):
        """Compute the convergence rate based on the two last errors and mesh sizes."""
        norm_list = list(self.errors.keys())[1:]    # exclude 'h'
        conv_rates = {key: None for key in norm_list}

        error_list = list(self.errors.items())[1:]
        for key, value in error_list:
            if np.all([val is None for val in value]):
                conv_rates[key] = np.nan
            else:
                p = (np.log(value[-1] / value[-2])) / np.log(
                    self.errors['h'][-1] / self.errors['h'][-2]
                )
                conv_rates[key] = p
        return conv_rates

    def save_results(self, folder_name : str, order_poly: str):
        """Compute convergence rates, format results, print, and save to files."""
        # 1: Compute convergence rates
        conv_rates = self.compute_convergence_rates()

        # 2: Format table
        table_str, table_data, headers = self._format_table(conv_rates)

        # 3: Print to terminal
        print(f"\nResults for {self.formulation} formulation ({order_poly}):")
        print(table_str)

        # 4: Create output directory [Global Env -> Formulation -> Polynomial degree]
        folder = os.path.join(folder_name, self.formulation, order_poly)
        os.makedirs(folder, exist_ok=True)

        # 5: Save to .txt
        txt_path = os.path.join(folder, f'table_{self.formulation}.txt')
        self._save_txt(table_str, txt_path)

        # 6: Save to .npy
        npy_path = os.path.join(folder, f'table_{self.formulation}.npy')
        data_dict = {
            'errors': self.errors,
            'conv_rates': conv_rates
        }
        self._save_npy(data_dict, npy_path)

        return table_str

    def _format_table(self, conv_rates: dict):
        headers = ['Norm'] + [f'{h}' for h in self.errors['h']] + ['Convergence Rate']
        table_data = []

        error_list = list(self.errors.items())[1:]
        for key, value in error_list:
            row = [key] + value + [conv_rates[key]]
            table_data.append(row)

        table_str = tabulate(table_data, headers=headers, tablefmt="github")
        return table_str, table_data, headers

    @staticmethod
    def _save_txt(table_str, txt_path):
        """Save the formatted table as a .txt file."""
        with open(txt_path, 'w') as f:
            f.write(table_str)

    @staticmethod
    def _save_npy(data_dict, npy_path: str):
        """Save the results dictionary as a .npy file."""
        np.save(npy_path, data_dict, allow_pickle=True)


    
    
    
