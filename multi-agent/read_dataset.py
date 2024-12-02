import pandas as pd

# Define file paths for the CERT dataset
logon_file = "r4.1/logon.csv"
file_access_file = "r4.1/file.csv"
email_file = "r4.1/email.csv"
device_file = "r4.1/device.csv"
http_file = "r4.1/http.csv"
psychometric_file = "r4.1/psychometric.csv"
ground_truth_file = "answers/r4.1-1.csv"

# Load dataset
logon_df = pd.read_csv(logon_file)
file_access_df = pd.read_csv(file_access_file)
email_df = pd.read_csv(email_file)
device_df = pd.read_csv(device_file)
http_df = pd.read_csv(http_file)
psychometric_df = pd.read_csv(psychometric_file)
ground_truth_df = pd.read_csv(ground_truth_file, names=["log_type", "id", "date", "user", "pc", "activity_or_url", "content"], header=None)

# Define the CERT data dictionary
data_dict = {
    'logon': logon_df,
    'file_access': file_access_df,
    'email': email_df,
    'device': device_df,
    'http': http_df
}