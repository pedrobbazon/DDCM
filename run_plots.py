import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg')
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


from ddcm_utils import *

FitForcheimerParams('myDATA.txt')