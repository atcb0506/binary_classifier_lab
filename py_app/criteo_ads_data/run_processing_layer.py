import argparse
import logging
import os

from config import CAT_COLUMNS
from layers.udf import adapting_preprocssing_layer
from dataprep.udf import dataprep


if __name__ == '__main__':

    # init logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_key', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    # init config
    feature_key = args.feature_key
    input_path = args.input_path
    output_path = args.output_path

    # create output dir
    dest_pathname = os.path.join(output_path, '')
    if not os.path.exists(os.path.dirname(dest_pathname)):
        os.makedirs(os.path.dirname(dest_pathname))
        logging.info(f'Created directory {dest_pathname}.')

    # load data
    data = dataprep(
        data_path=input_path,
        batch_size=512,
        test_split=False,
    )

    lst_feature = ['numeric']
    lst_feature.extend(CAT_COLUMNS)
    for feature_key in lst_feature:
        adapting_preprocssing_layer(
            feature=feature_key,
            data=data,
            output_path=dest_pathname
        )
