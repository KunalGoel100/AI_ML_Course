# Polinomial Regression Model
# Can make Linear, Quadratic, Cubit, etc lines

import matplotlib
from matplotlib import pyplot as plt

# Initial weights (Define the curve type)
W = [0,0,0]
# Learning Rate
a = 0.0001
# Stopping criteria
E = 0.1

# Data
data = [[1,1],[2,4],[3,9],[4,16],[5,25],[6,36],[7,49],[8,64],[9,81],[10,100]]
# data = [[1,2],[2,4],[3,6],[4,8],[5,10],[6,12]]
# # Normalize data
# x_max = max(d[0] for d in data)
# data = [[d[0]/x_max, d[1]] for d in data]

def GradientDescent(W):
    Guess = []
    for j in range(0,len(data),1):
        sum = 0
        for i in range(0,len(W),1):
            sum = sum + W[i]*data[j][0]**i
        Guess.append(sum)
    print(Guess)

    Cost = 0
    for i in range (0,len(data),1):
        Cost = Cost + ((data[i][1] - Guess[i])**2)/len(data)
    print(Cost)


    Grad = [0]*len(W)
    for i in range(0,len(Grad),1):
        for j in range (0,len(data),1):
            Grad[i] = Grad[i] - ((data[j][1] - Guess[j])*data[j][0]**i)/len(data)
    print(Grad)

    for i in range (0,len(W),1):
        W[i] = W[i] - a*Grad[i]
    return [W, Cost]

Cost = 1
count = 0
#################################
## Graph
plt.figure(figsize=[10,6])
# plt.subplot(2,2,1)


#################################
while Cost >= E:
# for i in range(0,10,1):
    [W, Cost] = GradientDescent(W)
    count = count +1
    if count >= 100000:
        break
    plt.plot(count,Cost,marker='o')

print(W)
print(count)
plt.show()
mean = 0
for i in data:
    mean = mean + i[1]
mean = mean/len(data)
print(f"mean = {mean}")

SST = 0
for i in data:
    SST = SST + (i[1] - mean)**2

SSE = Cost*len(data)
print(f"SSE = {SSE}")

print(f"SST = {SST}")
RSquare = 1 - SSE/SST

print(F"RSquare = {RSquare}")
####################






