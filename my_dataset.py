import random
list1 = [1,2,3,4,5,6,7] # 所有数据

batch_size = 2  # 一次拿多少个, 最后一个不够两个也会拿出来
epoch = 2  # 轮次, 背十轮每轮4次
shuffle = True  # 打乱


print(list1)

for e in range(epoch):
    if shuffle:
        random.shuffle(list1)
    for i in range(0, len(list1), batch_size): # 数据加载的过程
        batch_data = list1[i:i+batch_size]
        print(batch_data)
        # [1, 2]
        # [3, 4]
        # [5, 6]
        # [7]

class MyDataset:
    def __init__(self, all_datas, batch_size, shuffle=True):
        self.all_datas = all_datas
        self.batch_size = batch_size
        self.shuffle = shuffle
        pass


if __name__ == "__main__":
    all_datas = [1,2,3,4,5,6,7]
    batch_size = 2
    shuffle = True
    dataset = MyDataset(all_datas, batch_size, shuffle)

    for batch_data in dataset:
        print(batch_data)


