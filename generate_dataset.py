import random
import argparse
import tqdm
import os

def process(sentence):
    words = sentence.split(" ")
    new_sentence = ""
    for word in words:
        if random.random() < 0.5:
            i = random.randint(0,len(word))
            word = word[:i] + " " + word[i:]
    
        if random.random() < 0.3:
            new_sentence += word
        else:
            new_sentence += word + " "
    
    return new_sentence.strip()

def generate_dataset(filepath):
    print("Generating Data " + filepath)
    data = []
    with open(filepath) as file:
        for line in tqdm.tqdm(file):

            line = line.strip().split('\t')
            lines = line[0].split(". ")

            for org_line in lines:
                input_line = process(org_line)
                label_line = org_line
                data.append("\t".join([input_line, label_line]))
    
        file_name = os.path.basename(filepath)


    
    print("input")
    print(data[0].split("\t")[0])
    print("label")
    print(data[0].split("\t")[1])

    with open(f"data/{file_name}", "w") as file:
        file.write("\n".join(data))
            
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, required= True, help="path to the dataset")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    arg = arg_parse()
    generate_dataset(arg.filepath)

        


