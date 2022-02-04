import os
import pandas as pd
import variables

for file in os.listdir(variables.RESULTS_DIRECTORY):
    results_file_path = os.path.join(variables.RESULTS_DIRECTORY, file)
    race_data = pd.read_csv(results_file_path)
    print(race_data["place"][0])