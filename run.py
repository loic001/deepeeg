from experiments.base import ExpeRunner
import logging
import argparse

from experiments.base import ExperimentLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--times")
    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')

    parser.set_defaults(render=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    config = 'experiments.config_files.{}'.format(args.config)

    for i in range(int(args.times)):
        expe = ExpeRunner(config, render=args.render, suffix=i)
        expe.run()
        # print('runn')
        # logger = ExperimentLogger('/home/lolo/projects/crnl/deepeeg/results/eeg_net_sgd_weighted_filtered_iay/log', clear_on_load=False)
        # for k, v in logger.datas.items():
        #     print(v.keys())
