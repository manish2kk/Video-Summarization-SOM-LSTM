import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import os

# generate data points
n_dim = 2

n_c = 250
rand = np.random.RandomState(0)
c = rand.uniform(0, 1, (n_c, n_dim))

n_p = 1
patterns = np.array([[0]*n_dim])
for row in c:
    data_pt = np.random.normal((row), 0, (n_p, n_dim))
    patterns = np.concatenate((patterns, data_pt), axis=0)
patterns = patterns[1:]

# train SOM
w_width = 8
w_height = 8
prev_w = np.zeros((w_height, w_width, n_dim))
w = rand.uniform(0, 0.5, (w_width, w_height, n_dim))
radius0 = max(w_width,w_height)/2
learning_rate0 = 0.1
epochs = 250
radius=radius0
learning_rate = learning_rate0
max_iterations = len(patterns)+1
too_many_iterations = 10*max_iterations
convergence = [1]
timestep=1
e=0.0001
flag=0
epoch=0
print(patterns.shape)
index = 0

while epoch<epochs:
    print('epoch:', epoch)
    shuffle = np.random.randint(len(patterns), size=len(patterns))
    for i in range(len(patterns)):

        J = np.linalg.norm(w - prev_w)
        if  J <= e:
            flag=1
            break
        else:
            pattern = patterns[shuffle[i]]
            pattern_ary = np.tile(pattern, (w_height, w_width,1))
            Eucli_MAP = np.linalg.norm(pattern_ary - w, axis=2)
            BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
            prev_w = np.copy(w)

            for i in range(w_height):
                for j in range(w_width):
                    distance = np.linalg.norm([i - BMU[0], j - BMU[1]])
                    if distance <= radius:
                        w[i][j] = w[i][j] + learning_rate*(pattern-w[i][j])

            learning_rate = learning_rate0#*(1-(epoch/epochs))
            radius = radius*math.exp(-epoch/epochs)
            timestep+=1

        # Grid View
        if index%5==0:
            plt.figure()
            plt.scatter(patterns[:,0], patterns[:,1],s=10)
            wx = w[:,:,0]
            wy = w[:,:,1]
            plt.scatter(wx,wy,s=20)

            for i in range(w_height):
                x = []
                y = []
                for j in range(w_width):
                    x.append(w[i][j][0])
                    y.append(w[i][j][1])
                plt.plot(x,y)

            for i in range(w_height):
                x = []
                y = []
                for j in range(w_width):
                    x.append(w[j][i][0])
                    y.append(w[j][i][1])
                plt.plot(x,y)
            plt.savefig("plot/img"+str(index)+".png")
            plt.close()
        index+=1

    if J < min(convergence):
        print('Lower error found: %s' %str(J) + ' at epoch: %s' % str(epoch))
        print('\tLearning rate: ' + str(learning_rate))
        print('\tNeighbourhood radius: ' + str(radius))
        w_final = w

    convergence.append(J)
    if flag==1:
        break
    epoch+=1


# make vide

image_folder = 'plot'
video_name = 'video.avi'
images = [img for img in os.listdir(image_folder) if
          img.endswith(".png")]
temp = []
for name in images:
    temp.append([int(name[3:][:-4]), name])
temp = sorted(temp)
temp = np.array(temp)
images = temp[:,1]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 10, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
cv2.destroyAllWindows()
video.release()


# heat map
'''
BMU = np.zeros([2],dtype=np.int32)
result_map = np.zeros([w_height, w_width, n_dim],dtype=np.float32)
i=0

for pattern in patterns:
    pattern_ary = np.tile(pattern, (w_height, w_width, 1))
    Eucli_MAP = np.linalg.norm(pattern_ary - w_final, axis=2)
    BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
    x = BMU[0]
    y = BMU[1]
    for dim in range(n_dim):
        if classes[i] == 'Iris-setosa':
            if result_map[x][y][dim] <= 0.5:
                temp = [0]*n_dim
                temp[dim] = 0.5
                result_map[x][y] += np.asarray(temp)
    i+=1

result_map = np.flip(result_map,0)
print("Red = Iris-Setosa")
print("Blue = Iris-Virginica")
print("Green = Iris-Versicolor")
plt.imshow(result_map, interpolation='nearest')
plt.show()
'''
