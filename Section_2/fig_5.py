import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



file_name="results.csv"

# current dir 
current_dir = os.getcwd()
current_dir = os.path.join(current_dir, "Section_2")

# Load the CSV file into a DataFrame
file_path = os.path.join(current_dir, file_name)

data = pd.read_csv(file_path)
print(data)



# Set the index to the 'model_name' column to match the figures
data.set_index('model_name', inplace=True)

# Plot a heatmap
plt.figure(figsize=(18, 10))  # You can change the size to fit your needs
sns.heatmap(data, annot=True,fmt=".3f", cmap="crest")

# Add labels and a title if you want
plt.title('Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Models')

# Rotate the x-axis labels slightly and y-axis labels horizontally
plt.xticks(rotation=30, ha='right')  # Tilt the metric labels slightly
plt.yticks(rotation=0)  # Make model_name labels horizontal

# make folder png_files if not exist in Section_2/
if not os.path.exists(f'{current_dir}/png_files'):
    os.makedirs(f'{current_dir}/png_files')


# Save the figure
plt.savefig(f'{current_dir}/png_files/fig_5.png')

# Show the figure
# plt.show()