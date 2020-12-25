import argparse
import glob
import logging
import os
import time

from dataprep.tfrecord import tfrecord_writer

if __name__ == '__main__':

    # init logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    # init config
    input_path = args.input_path
    output_path = args.output_path
    input_files = glob.glob(f'{input_path}/*/*.txt')
    output_files = [
        x.replace('.txt', '.tfrecord.gz').replace(input_path, output_path)
        for x in input_files
    ]

    # write .tfrecord
    logging.info(f'Loaded: {input_files}')
    for i_file, o_file in zip(input_files, output_files):

        # create output dir
        if not os.path.exists(os.path.dirname(o_file)):
            os.makedirs(os.path.dirname(o_file))
            logging.info(f'Created directory {o_file}.')

        # write .tfrecord
        start_time = time.time()
        logging.info('Writing .tfrecord')
        tfrecord_writer(
            input_path=i_file,
            tfrecord_path=o_file,
            delimiter='\t',
        )
        end_time = time.time()

        logging.info(f'Finished turning {i_file} to {o_file} '
                     + f'spent: {round(end_time - start_time,2)}')
