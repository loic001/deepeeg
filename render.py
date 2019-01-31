import shelve

import glob
import os

from experiments.base import ExperimentLogger

if __name__ == '__main__':
    import dill as pickle
    import copy

    d = '/dycog/Jeremie/Loic/results/crnl_net_adam_weighted_filtered/'

    g = os.path.join(d, '*.dat')
    for f in glob.glob(g):
        experiment_logger = ExperimentLogger(f[:-4], overwrite=False)
        # expe.render_results(out_pdf_dir=d, all_epochs=False)
        # expe.render_results(out_pdf_dir=d, all_epochs=True)
        experiment_logger.render_results(out_pdf_dir=d)
