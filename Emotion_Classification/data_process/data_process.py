import pandas as pd


def get_csv_data(file_path, file_type, train_scale=0.6):
    """
        get csv data by file_type
        Args:
            file_type: return train_data, valid_data, test_data by file_type
    """
    assert file_type in ['train', 'valid', 'test']
    data = pd.read_csv(file_path, usecols=['content','sentiment_value'], encoding='UTF-8')
    data = data.rename(columns={'content':'text','sentiment_value':'label'})
    data = data.iloc[:200,:] # 取少量数据跑看下效果
    # 多分类将label进行one-hot encoding
    data['positivate'] = data['label'].apply(lambda x: 1 if x==1 else 0)
    data['neutral'] = data['label'].apply(lambda x: 1 if x==0 else 0)
    data['negative'] = data['label'].apply(lambda x: 1 if x==-1 else 0)
    data = data.drop('label', axis=1)
    # 获取train_data和valid_data
    train_index = int(len(data)*train_scale)
    test_index = int(len(data)*((1-train_scale)/2))
    train_data = data.iloc[:train_index , :]
    valid_data = data.iloc[train_index:-test_index , :]
    valid_data = valid_data.reset_index(drop=True) # 必须
    test_data  = data.iloc[-test_index: , :]
    test_data  = test_data.reset_index(drop=True)
    if file_type == 'train':

        return train_data

    if file_type == 'valid':

        return valid_data

    if file_type == 'test':

        return test_data


def txt_to_csv():
    pass


def json_to_csv():
    pass