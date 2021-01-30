#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import numpy as np

def pareto_frontier_multi(myArray):
    
    myArray=np.array(sorted(myArray, key=lambda x: x[0], reverse=True))
    
    pareto_frontier = myArray[0:1,:]
    potential=np.array([[0,0,0]])
    
    for row in myArray[1:,:]:
        if row[1] <= pareto_frontier[-1][1]:
            pareto_frontier = np.concatenate((pareto_frontier, [row]))
        else:
            potential=np.concatenate((potential, [row]))

    for row in potential[1:,:]:
        if sum([row[2] <= pareto_frontier[x][2]
                for x in range(len(pareto_frontier))]) == len(pareto_frontier):
            pareto_frontier = np.concatenate((pareto_frontier, [row]))

    return pareto_frontier

def test():
    myArray = [[16.67, 528, 40.55],[16.67, 549, 44.93],[93.33, 98, 18.72],[94.1,171,31.75],[98.51,232,46.99],
                    [98.14,92,11.12],[88.5,33,18.43],[94.44,57,23.67],[84.44,80,25.1],[99.63,88,23.68]]
    print (pareto_frontier_multi(myArray))

test()

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =[1,2,3,4,5,6,7,8,9,10]
y =[5,6,2,3,13,4,1,2,4,8]
z =[2,3,3,3,5,7,9,11,9,10]

ax.scatter(x, y, z, c='r', marker='o')

x =[4,4]
y =[1,1]
z =[3,7]

ax.scatter(x, y, z, c='r', marker='x')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
'''











