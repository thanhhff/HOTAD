import os

input_path = "test/"
output_path = "train/"

labels = ["Applying sunscreen", "Arm wrestling", "Assembling bicycle", "BMX", "Baking cookies", "Beer pong", "Blow-drying hair", "Playing ten pins", "Braiding hair", "Building sandcastles", "Calf roping", "Camel ride", "Canoeing", "Carving jack-o-lanterns", "Changing car wheel", "Cleaning sink", "Clipping cat claws", "Curling", "Cutting the grass", "Decorating the Christmas tree", "Doing a powerbomb", "Doing crunches", "Elliptical trainer", "Doing fencing", "Fun sliding down", "Futsal", "Gargling mouthwash", "Grooming dog", "Hand car wash", "Hanging wallpaper", "Hitting a pinata", "Hula hoop", "Hurling", "Ice fishing", "Installing carpet", "Kite flying", "Knitting", "Laying tile", "Longboarding", "Making a cake", "Making an omelette", "Mooping floor", "Painting fence", "Painting furniture", "Peeling potatoes", "Plastering", "Playing congas", "Playing drums", "Playing ice hockey", "Playing pool", "Playing rubik cube", "Powerbocking", "Putting in contact lenses", "Putting on shoes", "Raking leaves", "Riding bumper cars", "River tubing", "Rock-paper-scissors", "Rollerblading", "Roof shingle removal", "Rope skipping", "Running a marathon", "Scuba diving", "Sharpening knives", "Shuffleboard", "Skiing", "Slacklining", "Snow tubing", "Snowboarding", "Spread mulch", "Sumo", "Surfing", "Swinging at the playground", "Table soccer", "Throwing darts", "Using the rowing machine", "Wakeboarding", "Waxing skis", "Doing kickboxing", "Doing karate", "Tango", "Putting on makeup", "Playing bagpipes", "Cheerleading", "Clean and jerk", "Bathing dog", "Discus throw", "Playing field hockey", "Playing harmonica", "Playing saxophone", "Chopping wood", "Washing face", "Using the pommel horse", "Javelin throw", "Spinning", "Ping-pong", "Making a sandwich", "Brushing hair", "Playing guitarra", "Doing step aerobics", "Drinking beer", "Snatch", "Paintball", "Cleaning windows", "Brushing teeth", "Playing flauta", "Bungee jumping", "Triple jump", "Horseback riding", "Vacuuming floor", "Doing nails", "Washing hands", "Ironing clothes", "Using the balance beam", "Shoveling snow", "Tumbling", "Getting a tattoo", "Rock climbing", "Smoking hookah", "Shaving", "Getting a piercing", "Springboard diving", "Playing squash", "Playing piano", "Dodgeball", "Smoking a cigarette", "Sailing", "Getting a haircut", "Painting", "Shaving legs", "Hammer throw", "Skateboarding", "Polishing shoes", "Ballet", "Hand washing clothes", "Plataform diving", "Hopscotch", "Doing motocross", "Mixing drinks", "Starting a campfire", "Belly dance", "Volleyball", "Playing water polo", "Playing racquetball", "Kayaking", "Playing kickball", "Using uneven bars", "Washing dishes", "Pole vault", "Playing accordion", "Baton twirling", "Beach soccer", "Blowing leaves", "Bullfighting", "Capoeira", "Croquet", "Disc dog", "Drum corps", "Fixing the roof", "Having an ice cream", "Kneeling", "Making a lemonade", "Playing beach volleyball", "Playing blackjack", "Rafting", "Removing ice from car", "Swimming", "Trimming branches or hedges", "Tug of war", "Using the monkey bar", "Waterskiing", "Welding", "Drinking coffee", "Zumba", "High jump", "Wrapping presents", "Cricket", "Preparing pasta", "Grooming horse", "Preparing salad", "Playing polo", "Long jump", "Tennis serve with ball bouncing", "Layup drill in basketball", "Cleaning shoes", "Shot put", "Fixing bicycle", "Using parallel bars", "Playing lacrosse", "Cumbia", "Tai chi", "Mowing the lawn", "Walking the dog", "Playing violin", "Breakdancing", "Windsurfing", "Removing curlers", "Archery", "Polishing forniture", "Playing badminton"]

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
    