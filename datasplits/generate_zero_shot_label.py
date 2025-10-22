import json 

original_data_path = '../data/thumos/annotations/thumos14.json'
output_data_path = '../data/thumos/annotations/50-50/'

train_split_path = '../datasplits/thumos/50-50/train/split_cosine.list'
test_split_path = '../datasplits/thumos/50-50/test/split_cosine.list'
file_output_name = 'split_cosine.json'


def read_split_file(split_file_path):
    with open(split_file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

train_label = read_split_file(train_split_path)
test_label = read_split_file(test_split_path)

# print rate of train and test
print("Train rate: ", 1.0 * len(train_label) / (len(train_label) + len(test_label)))

with open(original_data_path, 'r') as f:
    data = json.load(f)

    # Calculate the number of videos in train and test
    train_count = 0
    test_count = 0
    ignore_count = 0

    for item in data['database']:
        subset = data['database'][item]['subset']
        annotations = data['database'][item]['annotations']
        
        check_in_train = False
        check_in_test = False
        action_list = []

        for annotation in annotations:
            if annotation['label'] in train_label:
                check_in_train = True
            if annotation['label'] in test_label:
                check_in_test = True

            if annotation['label'] not in action_list:
                action_list.append(annotation['label'])

        if check_in_train and check_in_test:
            ignore_count += 1
            data['database'][item]['subset'] = 'ignore'
        elif check_in_train:
            train_count += 1
            data['database'][item]['subset'] = 'validation'
        elif check_in_test:
            test_count += 1
            data['database'][item]['subset'] = 'test'

    print("Train count: ", train_count)
    print("Test count: ", test_count)
    print("Ignore count: ", ignore_count)

    with open(output_data_path + file_output_name, 'w') as f:
        json.dump(data, f)

    print("Done")



        



