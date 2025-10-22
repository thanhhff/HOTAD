import os

input_path = "test/"
output_path = "train/"

labels = ["BasketballDunk", "Billiards", "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "Diving", "FrisbeeCatch", "GolfSwing", "HighJump", "JavelinThrow", "LongJump", "Shotput", "TennisSwing", "VolleyballSpiking", "BaseballPitch", "HammerThrow", "PoleVault", "SoccerPenalty", "ThrowDiscus"]

def get_file_names(input_path):
    file_names = []
    for file in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path, file)):
            file_names.append(file)
    return file_names

file_names = get_file_names(input_path)

for file_name in file_names: 
    test_labels = []
    with open(input_path + file_name, 'r') as file:
        for line in file:
            label = line.strip()
            test_labels.append(label)
    
    # cerate train_labels 
    train_labels = list(set(labels) - set(test_labels))
    with open(output_path + file_name, 'w') as file:
        for label in train_labels:
            file.write(label + "\n")
    