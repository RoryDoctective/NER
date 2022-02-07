
def remove_title():
    data_dir = 'Dataset/Finance/test_finance_all_title.txt'
    output_dir = 'Dataset/Finance/demo.test'

    data = []

    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:  # for each line
            if line[:2] == '<t':
                print(line)
                continue
            else:
                data.append(line.strip('\n'))

    with open(output_dir, 'w', encoding='utf-8') as f:
        for i in range(len(data)):
            f.write(data[i]+'\n')



data_dir = 'Dataset/E-commerce/original_demo.dev'
output_dir = 'Dataset/E-commerce/demo.dev'

data = []

with open(data_dir, 'r', encoding='utf-8') as f:
    for line in f:  # for each line
        if line.strip('\n') == ' \tO':
            print(line)
            continue
        else:
            # print(line)
            data.append(line.strip('\n'))

with open(output_dir, 'w', encoding='utf-8') as f:
    for i in range(len(data)):
        f.write(data[i] + '\n')
