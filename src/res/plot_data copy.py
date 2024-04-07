
import seaborn as sns
import pandas as pd
import matplotlib
import os
from statistics import mean 

log_folder = "./logs/24000"
data = dict()
methods = {0: "getrf", 1: "cudaGraph", 2: "ours", 3: "MgGetrf", 4: "StarPU"}

def getGFLOPs(n, time):
    flop = (2.0*float(n*n*n))/3.0
    return flop/time #/1000.0

def readStarPU(file, params):
    method = methods[int(params[0][3:])]
    size = int(params[1])
    gpu_count = int(params[3][:-3])
    data[method] = data.get(method,dict())
    data[method][gpu_count] = data[method].get(gpu_count,[])
    res = data[method][gpu_count]
    max_runtime = 0.0
    for line in file.readlines():
        if (line.startswith(str(size))):
            print(line)
            runtime = float(line.split("	")[1])/1000.0
            res.append(runtime)
    print(data[method][gpu_count])
    data[method][gpu_count] = mean(data[method][gpu_count])

def readMG(file, params):
    method = methods[int(params[0][3:])]
    size = int(params[1])
    gpu_count = int(params[3][:-3])
    data[method] = data.get(method,dict())
    data[method][gpu_count] = data[method].get(gpu_count,[])
    res = data[method][gpu_count]
    max_runtime = 0.0
    for line in file.readlines():
        if (line.startswith("Total")):
            print(line)
            runtime = float(line.split(" ")[-1])
            res.append(runtime)
    print(data[method][gpu_count])
    data[method][gpu_count] = mean(data[method][gpu_count])

def readOurs(file, params):
    method = methods[int(params[0][3:])]
    size = int(params[1])
    gpu_count = int(params[3][:-3])
    data[method] = data.get(method,dict())
    data[method][gpu_count] = data[method].get(gpu_count,[])
    res = data[method][gpu_count]
    gpu_idx = 0
    max_runtime = 0.0
    for line in file.readlines():
        if (line.startswith("device")):
            print(line)
            runtime = float(line.split(" ")[-1])
            max_runtime = max(runtime, max_runtime) 
            gpu_idx += 1
            if (gpu_idx == gpu_count):
                res.append(max_runtime)
                max_runtime = 0.0
                gpu_idx = 0
    print(data[method][gpu_count])
    data[method][gpu_count] = mean(data[method][gpu_count])

def readData():
    for f in os.listdir(log_folder):
        filename = os.fsdecode(f)
        print(filename)
        if filename.endswith(".log"): 
            file = open(os.path.join(log_folder, filename), "r")
            # print(os.path.join(directory, filename))
            spl_name = filename[:-4].split("_")
            method = int(spl_name[0][3:])
            # size = int(spl_name[1])
            # print(method)
            # print(size)
            # if len(spl_name) > 2:
            #     tiles = int(spl_name[2])
            #     print(tiles)
            # if len(spl_name) > 3:
            #     gpu_count = int(spl_name[3][:-3])
            #     print(gpu_count)
            # else:
            if (method < 3):  
                if (method == 0):
                    spl_name.append("1")
                if (method <= 1):
                    spl_name.append("1GPU")
                readOurs(file, spl_name)
            if (method == 3):
                readMG(file, spl_name)
            if (method == 4):
                readStarPU(file, spl_name)
            continue
        else:
            continue 

readData()

sns.set_theme(style="whitegrid")

penguins = sns.load_dataset("penguins")
print(penguins)

df = pd.DataFrame.from_dict(data).transpose()
df.index.name = 'method'
df.reset_index(inplace=True)
print(df)
# g = sns.catplot(
#     data=df, kind="bar",
#     x="species", y="body_mass_g", hue="sex",
#     errorbar="sd", palette="dark", alpha=.6, height=6
# )
g = sns.catplot(
    data=df, kind="bar",
    x="method", y=
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
#g.legend.set_title("")
matplotlib.pyplot.show()