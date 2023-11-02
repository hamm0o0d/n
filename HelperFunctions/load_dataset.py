
import pandas as pd
from HelperFunctions.min_max_normalizer import MinMaxNormalizer

def load_dataset():
    df = pd.read_excel('Dry_Bean_Dataset.xlsx')

    indexes_with_nan = df[df['MinorAxisLength'].isnull()].index

    for rowIndex in indexes_with_nan:
        valueBefore = df['MinorAxisLength'][rowIndex - 1]
        valueAfter = df['MinorAxisLength'][rowIndex + 1]
        mean = (valueAfter + valueBefore) / 2
        df.at[rowIndex, 'MinorAxisLength'] = mean


    areaNormalizer = MinMaxNormalizer()
    perimNormalizer = MinMaxNormalizer()
    majorNormalizer = MinMaxNormalizer()
    minorNormalizer = MinMaxNormalizer()
    
    df['Area'] = areaNormalizer.fit_transform(df['Area'])
    df['Perimeter'] = perimNormalizer.fit_transform(df['Perimeter'])
    df['MajorAxisLength'] = majorNormalizer.fit_transform(df['MajorAxisLength'])
    df['MinorAxisLength'] = minorNormalizer.fit_transform(df['MinorAxisLength'])

    return df