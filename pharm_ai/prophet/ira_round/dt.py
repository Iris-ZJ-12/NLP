import pandas as pd
from pharm_ai.util.utils import Utilfuncs as u
import os.path

def dt20201124():
    x = '轮次分类补充2011123-缩轮次+标题 -交付.xlsx'
    df = pd.read_excel(x, '1')
    titles = df['title'].tolist()
    paras = df['paragraph'].tolist()
    target_text = df['round'].tolist()
    input_text = []
    for p, t in zip(paras, titles):
        txt = f'title: {str(t)} paragraph: {str(p)}'
        input_text.append(txt)

    df = pd.DataFrame({'input_text': input_text, 'target_text': target_text})
    df = df.sample(frac=1, random_state=123)
    train = df.iloc[80:]
    test = df.iloc[:80]
    h5 = 'train_test-20201124-2.h5'
    train.to_hdf(h5, 'train')
    test.to_hdf(h5, 'test')


def train_test():
    h5 = os.path.join(os.path.dirname(__file__),'train_test-20201124-2.h5')
    train = pd.read_hdf(h5, 'train')
    print(len(train))
    test = pd.read_hdf(h5, 'test')
    print(len(test))
    return train, test

def dt20201125():
    x = 'labels.xlsx'
    labels = pd.read_excel(x, '1')['labels'].tolist()[::-1]
    print(labels)

if __name__ == '__main__':
    train_test()