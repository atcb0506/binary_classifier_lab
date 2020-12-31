import argparse
import logging
import subprocess

from comm_lib.s3_ops import download_dir


def main(
        bucket: str,
        prefix: str,
        local_logdir: str,
) -> None:

    # init
    tf_logdir = f'{local_logdir}/{prefix}'

    # clear log
    subprocess.run(
        ['rm', '-rf', tf_logdir],
    )

    # download tensorboard log
    download_dir(
        bucket=bucket,
        prefix=prefix,
        local_logdir=local_logdir,
    )

    # run tensorboard
    subprocess.run(
        ['tensorboard', '--logdir', tf_logdir],
    )


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--local_logdir', type=str)
    args = parser.parse_args()

    main(
        bucket=args.bucket,
        prefix=args.prefix,
        local_logdir=args.local_logdir,
    )
