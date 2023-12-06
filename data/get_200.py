import pandas as pd

file_path = "angles/goddess_angles.csv"
data = pd.read_csv(file_path)
print(data.head())
# Shuffle the DataFrame (since data/poseangles.generated_csv has each pose sequentially)
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# Select 200 random data points
sample_size = 200
random_sample = shuffled_data.head(sample_size)

print(random_sample.head())  # Optional: Print the first few rows of the random sample

# Save the random sample to a new CSV file
output_file = '200_angles/random_sample_200_goddess.csv'
random_sample.to_csv(output_file, index=False)

