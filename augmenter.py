# Dependencies
import pandas as pd
import numpy as np

# Load raw training data
colnames = pd.array(range(1,112321)) # The number of time intervals in the database
train_controls = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_controls_data.csv', names = colnames, header = None)
train_cases = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_cases_data.csv', names = colnames, header = None)

# Augment
def chunker(data):
    output = pd.DataFrame()
    for i in np.arange(len(data)):
        chunk_1 = data.iloc[i,0:17280]
        chunk_2 = data.iloc[i,17280:34560]
        chunk_3 = data.iloc[i,34560:51840]
        chunk_4 = data.iloc[i,51840:69120]
        chunk_5 = data.iloc[i,69120:86400]
        chunk_6 = data.iloc[i,86400:103680]
        chunk_7 = data.iloc[i,103680:112320]
        order = ['chunk_1','chunk_2','chunk_3','chunk_4','chunk_5','chunk_6','chunk_7']
        np.random.shuffle(order)
        key = {'chunk_1':chunk_1,'chunk_2':chunk_2,'chunk_3':chunk_3,'chunk_4':chunk_4,'chunk_5':chunk_5,'chunk_6':chunk_6,'chunk_7':chunk_7}
        augmented = pd.DataFrame(np.concatenate([key[order[0]],key[order[1]],key[order[2]],key[order[3]],key[order[4]],key[order[5]],key[order[6]]]))
        output = pd.concat([output,augmented],axis=1)
        if i % 50 == 0:
            print(f"Just finished augmenting row {i} out of {len(data)}!")
    return(output)

# Augment train data
train_controls_augmented = chunker(data=train_controls)
train_cases_augmented = chunker(data=train_cases)

# Invert both to harmonize format with other data
train_controls_augmented = pd.DataFrame.transpose(train_controls_augmented)
train_cases_augmented = pd.DataFrame.transpose(train_cases_augmented)

# Write out
train_controls_augmented.to_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_controls_data_augmented.csv',header=False)
train_cases_augmented.to_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_cases_data_augmented.csv',header=False)