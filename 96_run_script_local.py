import subprocess
import threading

def stream_output(stream, prefix=''):
    for line in stream:
        print(prefix + line, end='')

# Define the path to the Python script and its arguments
home_dir = '/Users/kate_fieseler/PycharmProjects/edboplus_scope'
script_path = home_dir + '/test/thompson_SNAr/_benchmark.py'

# Create a list of arguments
args = [
    '--seed', '1',
    '--data', home_dir+'/test/thompson_SNAr/figure_S2/data/figure_S2_tidy.csv',
    '--batch', '96',
    '--rxn_components', home_dir+'/test/thompson_SNAr/figure_S2/reaction_components.json',
    '--rounds', '10',
    '--max_path', home_dir+'/test/thompson_SNAr/figure_S2/data/figure_S2_tidy_max_peak_height.csv'
]

# Run the script using subprocess.Popen()
process = subprocess.Popen(['python', script_path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Use threads to handle stdout and stderr in real-time
stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,))
stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "Error: "))

stdout_thread.start()
stderr_thread.start()

stdout_thread.join()
stderr_thread.join()

# Wait for the process to finish
process.wait()
