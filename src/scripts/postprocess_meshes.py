import argparse
import os
import os.path as osp
import subprocess

from src.utils.common import get_file_paths


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_dir", required=True)
    arg_parser.add_argument("--file_type", required=True)
    args = arg_parser.parse_args()

    paths = get_file_paths(args.input_dir, file_type=args.file_type)

    for path in paths:
        """
        meshlabserver -i $IN_PATH -o $OUT_PATH -s /home/kejie/repository/fast_sdf/clean_mesh.mlx
        """
        print(path)
        out_path = path[:-4] + "_post_process.ply"
        commands = f"meshlabserver -i {path} -o {out_path} -s ./clean_mesh.mlx"
        commands = commands.split(" ")
        try:
            subprocess.run(commands, check=True)
        except subprocess.CalledProcessError:
            import pdb
            pdb.set_trace()


if __name__ == "__main__":
    main()