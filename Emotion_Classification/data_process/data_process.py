import pandas as pd


"""
    情感二分类处理方式
"""
def get_two_classes_csv_data(file_path, file_type, train_scale=0.6):
    """
        使用weibo_senti_100k二分类评论数据集
    """
    assert file_type in ['train', 'valid', 'test']
    data = pd.read_csv(file_path, encoding='UTF-8')
    data = data.rename(columns={'review':'text'})  
    data = data[['text','label']]
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


"""
    情感多分类数据处理方式
"""
def get_n_classes_csv_data(file_path, file_type, train_scale=0.6):
    assert file_type in ['train', 'valid', 'test']
    data = pd.read_csv(file_path, usecols=['content','sentiment_value'], encoding='UTF-8')
    data = data.rename(columns={'content':'text','sentiment_value':'label'})
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