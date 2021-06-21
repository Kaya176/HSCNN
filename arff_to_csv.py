'''
conver arff file to csv(or txt)
'''
import pandas as pd

def arff_to_csv(file_name):

    Class_idx = [str(idx) for idx in range(49060,49082)] #class01 ~ class 22
    if file_name.find(".arff") < 0 : 
        print("It is not arff file!!")
        return -1
    
    f = open(file_name)
    lines = f.readlines()
    attr = dict()
    Flag = False
    text = [] #result -> text
    labels = []

    for idx,line in enumerate(lines):
        if Flag: #Data part
            line = line.strip()[1:-1].split(sep = ',')
            if len(line) > 5:
                converted = []
                label = []
                Onece = True
                for l in line:
                    element = attr[l.split()[0]]
                    if l.split()[0] in Class_idx:
                        label.append(element[-2:])
                    elif element in ["_","__","___"] and Onece:
                        converted = []
                        Onece = False
                    else:
                        converted.append(element)
                #print(label)
                text.append(" ".join(converted).lower()) #add lower
                labels.append(" ".join(label))

        else: # Attribute part
            line = line.strip().split()
            if len(line) != 0:
                attribute = line[0]
                if attribute == "@attribute":
                    idx = idx-2
                    attr[str(idx)] = line[1]
                elif attribute == "@data":
                    Flag = True

    return pd.DataFrame({"Text" : text,"Label" : labels})

if __name__ == "__main__":
    train = arff_to_csv("tmc2007-train.arff")
    test = arff_to_csv("tmc2007-test.arff")

    train.to_csv("tmc2007-train.csv",index = False)
    test.to_csv("tmc2007-test.csv",index = False)