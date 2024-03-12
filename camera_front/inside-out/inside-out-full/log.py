import subprocess
import time
import os

# Get the current directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the path to the log file
log_file = os.path.join(script_dir, "temperature_log.txt")

# Function to get the current temperature
def get_temperature():
    # Run vcgencmd measure_temp command and capture output
    temperature = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("utf-8")
    # Extract temperature value
    temperature = temperature.split("=")[1].split("'")[0]
    return temperature

# Function to get the current core frequency
def get_core_frequency():
    # Run vcgencmd get_config arm_freq command and capture output
    frequency_output = subprocess.check_output(["vcgencmd", "get_config", "arm_freq"]).decode("utf-8")
    # Extract frequency value
    frequency = frequency_output.split("=")[1]
    return frequency

# Main function to continuously log temperature and core frequency readings
def log_temperature(interval_seconds=2):
    while True:
        # Get current temperature
        temperature = get_temperature()
        # Get current core frequency
        core_frequency = get_core_frequency()
        print(f'{temperature}°')
        print(f'{core_frequency} mhz')
        # Get current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Format log entry
        log_entry = f"{timestamp}: Temperature = {temperature} C, Core Frequency = {core_frequency} Hz\n"
        # Append log entry to the log file
        with open(log_file, "a") as file:
            file.write(log_entry)
        # Wait for the specified interval before logging again
        time.sleep(interval_seconds)

# Call the main function to start logging temperatures and core frequencies
log_temperature()
