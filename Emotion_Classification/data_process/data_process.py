import pandas as pd


def get_csv_data(file_path, train_scale=0.7):
    """
        get train_data and valid_data
        Args:
            train_scale: proportion of train_data
    """
    data = pd.read_csv(file_path, usecols=['content','sentiment_value'], encoding='UTF-8')
    data = data.rename(columns={'content':'text','sentiment_value':'label'})
    # 多分类将label进行one-hot encoding
    data['positivate'] = data['label'].apply(lambda x: 1 if x==1 else 0)
    data['neutral'] = data['label'].apply(lambda x: 1 if x==0 else 0)
    data['negative'] = data['label'].apply(lambda x: 1 if x==-1 else 0)
    data = data.drop('label', axis=1)
    # 获取train_data和valid_data
    train_index = int(len(data)*train_scale)
    train_data = data.iloc[:train_index , :]
    valid_data = data.iloc[train_index: , :]
    valid_data = valid_data.reset_index() # 必须

    return train_data, valid_data


def txt_to_csv():
    pass


def json_to_csv():
    pass