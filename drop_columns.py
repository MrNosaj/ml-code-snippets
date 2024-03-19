# List of columns to drop
columns_to_drop = ['Cabin', 'AgeGroup', 'Deck', 'Ticket','FareRange','Title','Name','SibSp','Parch','Fare','Embarked']

# Drop columns from the train data if they exist
for column in columns_to_drop:
    if column in train_data.columns:
        train_data.drop(columns=[column], inplace=True)
