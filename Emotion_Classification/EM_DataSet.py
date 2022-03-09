from torch.utils.data import Dataset

# 定义数据集
class Emotion_Dataset(Dataset):
    """
        按照原来的思路：text是一个dataset、label又是一个dataset,zip到一起后传入DataLoader
        现在可以给Dataset两个返回值,一个是text,一个是label
    """
    def __init__(self, csv_file, n_class):
        self.dataset = csv_file
        self.n_class = n_class
        if n_class == 2:
            self.text = self.dataset['text']
            self.label = self.dataset['label']
        else:
            """
                根据具体多情感数据集而定label
            """
            self.text = self.dataset['text']
            self.label = self.dataset[['positivate','neutral','negative']].values

    
    def __len__(self):
        
        return len(self.dataset)

    
    def __getitem__(self, idx):

        text = self.text[idx]
        label = self.label[idx]

        return (text, label)
