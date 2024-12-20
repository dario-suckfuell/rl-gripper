import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Define the folder containing CSV files
csv_files = glob.glob("*.csv")


# Get a list of all CSV files in the folder
#csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Set the smoothing window size
window_size = 200  # Adjust the window size as needed

for file_name in csv_files:
    # Read the CSV file
    data = pd.read_csv(file_name)

    # Apply smoothing (using a simple moving average)
    data['Smoothed_Value'] = data['Value'].rolling(window=window_size, min_periods=1).mean()

    # Save the smoothed data to a new file
    smoothed_file_name = file_name.replace('.csv', '_smoothed.csv')
    data.to_csv(smoothed_file_name, index=False)

    # Plot Step vs Smoothed_Value
    plt.figure()
    plt.plot(data['Step'], data['Smoothed_Value'], label='Smoothed Value')
    plt.xlabel('Step')
    plt.ylabel('Smoothed Value')
    plt.title(f"Step vs Smoothed Value for {os.path.basename(file_name)}")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Smoothed data saved to: {smoothed_file_name}")

print("Smoothing applied to all files in the folder.")

