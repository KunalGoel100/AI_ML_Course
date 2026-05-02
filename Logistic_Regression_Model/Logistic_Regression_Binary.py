import math
# Initial weights (Define the curve type)
W = [0,5,20]
# Learning Rate
a = 0.1
# Stopping criteria
ee = 0.01

# Data
# data = [[-1,-2,0],[3,1,1],[-10,-23,0],[-12,-23,0],[4,3,1],[-56,-45,0],[23,54,1]]
# AND Gate
data1 = [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
# NOR Gate
data = [[0,0,1],[0,1,0],[1,0,0],[1,1,0]]
def Logistic(W,data):
    Z = []
    X = []
    for i in data:
        X.append([1,i[0],i[1]])

    # print(f"X {X}")
    for i in data:
        Z.append(W[0] + W[1]*i[0] + W[2]*i[1])

    print(f"Z = {Z}")

    Sigmoid = []
    for i in Z:
        Sigmoid.append((1/(1+math.exp(-i))))

    # Epsilon clipping
    E = 1e-15
    guess = []
    for i in Sigmoid:
        if i >= 1-E:
            guess.append(1-E)
        elif i <= E:
            guess.append(E)
        else:
            guess.append(i)
    print(f"Guess {guess}")


    Cost = []
    for i in range(0,len(guess),1):

            Cost.append(-(data[i][2]*math.log(guess[i]) + (1-data[i][2])*math.log((1-guess[i]))))

    Cost = sum(Cost)/len(data)
    print(f"Cost {Cost}")

    Grad = [0]*len(X[0])
    for i in range(0,len(X[0]),1):
        for j in range(0,len(data),1):
            Grad[i] = Grad[i] -((data[j][2] - guess[j])*X[j][i])

    for i in range(0,len(W),1):
        W[i] = W[i] - a*Grad[i]

    # print(W)
    return [W,Cost,guess]

# for i in range(0,100,1):
Cost = 1
count = 0

while Cost >= ee:
    count += 1
    [W,Cost,guess] = Logistic(W,data)
    if count >= 10000:
        break
print(f"Final W = {W}")






