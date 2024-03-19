missing_values = train_data.isnull().sum()
print("Columns with missing values for TRAIN data:")
print(missing_values[missing_values > 0])
print(" ")
