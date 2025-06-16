import os
import subprocess
import sys
import platform

MAX_RUNNING_TIME = "605s"

class TimeoutError(Exception):
    """Custom exception for timeout errors."""
    pass


def run_benchmark(model_file, input_folder, output_folder, log_file):
    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if platform.system() == "Darwin":
        timeout_command = "gtimeout"
    else:
        timeout_command = "timeout"

    # Get the path to the JAR file
    jar_path = os.path.join(model_file, "target", "ChallengeSBPO2025-1.0.jar")

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            print(f"\nRunning {filename}")
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

            # Main Python command
            cmd = [
                timeout_command,
                MAX_RUNNING_TIME,
                "python",
                model_file,
                input_file,
                output_file,
                log_file
            ]

            result = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()  # Set working directory directly
            )

            try:
                # Check for timeout (return code 124 is the standard timeout exit code)
                if result.returncode == 124:
                    error_msg = f"Execution timed out after {MAX_RUNNING_TIME} for {input_file}"
                    print(error_msg)
                    raise TimeoutError(error_msg)
                elif result.returncode != 0:
                    print(f"Execution failed for {input_file}:")
                    print(result.stderr)
                    raise RuntimeError(f"Execution failed for {input_file}: {result.stderr}")
            except TimeoutError as e:
                print(e)
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                # raise e


if __name__ == "__main__":
    if len(sys.argv) == 4:
        print("Usage: python run_challenge.py <model_file> <input_folder> <output_folder> <log_file>")

        model_file = sys.argv[1]
        input_folder = sys.argv[2]
        output_folder = sys.argv[3]
        log_file = sys.argv[4]
    else:
        model_file = os.path.join("analysis", "parametric.py")
        input_folder = os.path.join("datasets", "b")
        output_folder = os.path.join("output_challenge")
        log_file = os.path.join("analysis", "parametric_results.log")

    # Convert to absolute paths
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    run_benchmark(model_file, input_folder, output_folder, log_file)
