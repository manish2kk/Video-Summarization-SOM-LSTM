import numpy as np
import math
import random
from PIL import Image
import matplotlib.pyplot as plt

patterns = []
classes = []
filename = 'iris.txt'
file = open(filename,'r')

for line in file.readlines():
    row = line.strip().split(',')
    patterns.append(row[0:4])
    classes.append(row[4])
print("Iris Data Loaded")
file.close
patterns = np.asarray(patterns,dtype=np.float32)
sample_no = np.random.randint(0,len(patterns))
print("Sample pattern: " + str(patterns[int(sample_no)]))
print("Class of the above pattern: " + str(classes[int(sample_no)]))


def mapunits(input_len,size='small'):
    #print(input_len, size)
    heuristic_map_units = 5*input_len**0.54321
    if size == 'big':
        heuristic_map_units = 4*(heuristic_map_units)
    else:
        heuristic_map_units = 0.25*(heuristic_map_units)
    return heuristic_map_units

map_units = mapunits(len(patterns),size='big')
print("Heuristically computed appropriate no. of map units: "+str(int(map_units)))


def Eucli_dists(MAP,x):
    x = x.reshape((1,1,-1))
    #print(x)
    Eucli_MAP = MAP - x
    Eucli_MAP = Eucli_MAP**2
    Eucli_MAP = np.sqrt(np.sum(Eucli_MAP,2))
    return Eucli_MAP

input_dimensions = 4
map_width = 3
map_height = 3
prev_MAP = np.zeros((map_height,map_width,input_dimensions))
MAP = np.random.uniform(size=(map_height,map_width,input_dimensions))
radius0 = max(map_width,map_height)/2
learning_rate0 = 0.1
coordinate_map = np.zeros([map_height,map_width,2],dtype=np.int32)
#print('\nMAP\n',MAP,'\nprev_MAP\n', prev_MAP, '\nradius0\n', radius0, '\ncoordinate_map\n', coordinate_map)

for i in range(0,map_height):
    for j in range(0,map_width):
        coordinate_map[i][j] = [i,j]

epochs = 150
radius=radius0
learning_rate = learning_rate0
max_iterations = len(patterns)+1
too_many_iterations = 10*max_iterations
convergence = [1]
timestep=1
e=0.001
flag=0
epoch=0

images = []
while epoch<epochs:
    print('epoch:', epoch)
    shuffle = np.random.randint(len(patterns), size=len(patterns))
    for i in range(len(patterns)):

        J = np.linalg.norm(MAP - prev_MAP)
        if  J <= e: #if converged (convergence criteria)
            flag=1
            break
        else:
            pattern = patterns[shuffle[i]]
            pattern_ary = np.tile(pattern, (map_height, map_width,1))
            print(pattern,'\n',pattern_ary)
            exit()
            Eucli_MAP = np.linalg.norm(pattern_ary - MAP, axis=2)
            BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
            prev_MAP = np.copy(MAP)
    
            for i in range(map_height):
                for j in range(map_width):
                    distance = np.linalg.norm([i - BMU[0], j - BMU[1]])
                    if distance <= radius:
                        MAP[i][j] = MAP[i][j] + learning_rate*(pattern-MAP[i][j])

            learning_rate = learning_rate0*(1-(epoch/epochs))
            radius = radius0*math.exp(-epoch/epochs)
            timestep+=1

    if J < min(convergence):
        print('Lower error found: %s' %str(J) + ' at epoch: %s' % str(epoch))
        print('\tLearning rate: ' + str(learning_rate))
        print('\tNeighbourhood radius: ' + str(radius))
        MAP_final = MAP

    convergence.append(J)
    if flag==1:
        break
    epoch+=1


plt.plot(convergence)
plt.ylabel('error')
plt.xlabel('epoch')
plt.grid(True)
plt.yscale('log')
plt.show()
print('Number of timesteps: ' + str(timestep))
print('Final error: ' + str(J))

BMU = np.zeros([2],dtype=np.int32)
result_map = np.zeros([map_height,map_width,3],dtype=np.float32)
i=0
for pattern in patterns:
    pattern_ary = np.tile(pattern, (map_height, map_width, 1))
    Eucli_MAP = np.linalg.norm(pattern_ary - MAP_final, axis=2)
    # Get the best matching unit(BMU) which is the one with the smallest Euclidean distance
    BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
    x = BMU[0]
    y = BMU[1]
    if classes[i] == 'Iris-setosa':
        if result_map[x][y][0] <= 0.5:
            result_map[x][y] += np.asarray([0.5,0,0])
    elif classes[i] == 'Iris-virginica':
        if result_map[x][y][1] <= 0.5:
            result_map[x][y] += np.asarray([0,0.5,0])
    elif classes[i] == 'Iris-versicolor':
        if result_map[x][y][2] <= 0.5:
            result_map[x][y] += np.asarray([0,0,0.5])
    i+=1

result_map = np.flip(result_map,0)
#print result_map
print("Red = Iris-Setosa")
print("Blue = Iris-Virginica")
print("Green = Iris-Versicolor")
plt.imshow(result_map, interpolation='nearest')
plt.show()