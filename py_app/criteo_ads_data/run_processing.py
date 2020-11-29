import argparse
import logging
import time

from dataprep.tfrecord import tfrecord_writer

if __name__ == '__main__':

    # init logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default='train.txt')
    parser.add_argument('--data_path', type=str,
                        default='../../data/criteo_ads_data/dac')
    parser.add_argument('--nrow', type=int, default=10000)
    args = parser.parse_args()

    # init config
    data_filename = args.data_filename
    data_path = args.data_path
    nrow = args.nrow
    input_path = f'{data_path}/input/{data_filename}'
    output_path = f'{data_path}/output/tfrecord_{nrow}.tfrecord'

    # write .tfrecord
    start_time = time.time()
    logging.info('Writing .tfrecord')
    tfrecord_writer(
        input_path=input_path,
        tfrecord_path=output_path,
        delimiter='\t',
        nrow=nrow,
    )
    end_time = time.time()
    logging.info('Finished writing .tfrecord - '
                 + f'spent: {round(end_time - start_time,2)}')
