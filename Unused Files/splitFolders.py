import splitfolders

# Path to the folder you want to split
input_folder = "./Dataset"
# Output folder where the split data will be saved
output_folder = "./NewData"

# Ratio to split the data, e.g., 0.8 for 80% training and 0.2 for 20% testing
split_ratio = (0.8, 0.1, 0.1)

# Use split-folders to split the data
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=split_ratio, group_prefix=None)

print("Done !!")