import os
import subprocess
import sys
import platform
import traceback


def run_check_challenge(input_folder, output_folder):

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            print("="*50)
            print(f"Checking {filename}")
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)

            # Main Java command
            cmd = [
                "python",
                "checker.py",
                input_file,
                output_file
            ]

            result = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                text=True,
                # cwd="/"  # Set working directory directly
            )

            if result.returncode != 0:
                print(f"Execution failed for {input_file}:")
                print(traceback.format_exc())
                continue


if __name__ == "__main__":

    input_folder = os.path.abspath("datasets/b/")
    output_folder = os.path.abspath("output_challenge/")

    run_check_challenge(input_folder, output_folder)
