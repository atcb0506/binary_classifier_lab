"""
this is created for local splitting csv files,
this is not included in the any pipeline, and executed manually

"""
import argparse
import csv
import logging
import os


def split_csv(
        input_filepath: str,
        output_dir: str,
        delimiter: str,
        row_per_file: int,
        n_files: int = None,
        train_test_split: int = 5,
) -> None:

    # init
    base_output_filename = 'csv_data'
    if n_files is None:
        n_files = 9999
    file_idx = 0
    reader_pointer = 0
    source_eof = False

    with open(input_filepath, 'r') as rf:
        csv_reader = csv.reader(rf, delimiter=delimiter)

        while True:
            sub_dir = 'train'
            if file_idx % train_test_split == 0:
                sub_dir = 'test'

            output_filepath = f'{output_dir}/{sub_dir}/' \
                              f'/{base_output_filename}_{str(file_idx)}.txt'
            if not os.path.exists(os.path.dirname(output_filepath)):
                os.makedirs(os.path.dirname(output_filepath))
                logging.info(f'Created directory {output_filepath}.')

            with open(output_filepath, 'w') as wf:
                eof_pointer = reader_pointer + row_per_file
                print(f'reader_pointer: {reader_pointer}; '
                      f'eof_pointer: {eof_pointer}')
                for _ in range(reader_pointer, eof_pointer):
                    row_date = next(csv_reader, None)
                    if row_date is None:
                        source_eof = True
                        break
                    csv_writer = csv.writer(wf, delimiter=delimiter)
                    csv_writer.writerow(row_date)

                file_idx += 1
                reader_pointer = eof_pointer
                if source_eof or file_idx >= n_files:
                    break


if __name__ == '__main__':

    # args variable
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--delimiter',
                        type=lambda d: '\t' if d == '\\t' else d)
    parser.add_argument('--row_per_file', type=int,)
    parser.add_argument('--n_files', type=int, default=None)
    args = parser.parse_args()

    # run program
    split_csv(
        input_filepath=args.input_filepath,
        output_dir=args.output_dir,
        delimiter=args.delimiter,
        row_per_file=args.row_per_file,
        n_files=args.n_files,
    )
