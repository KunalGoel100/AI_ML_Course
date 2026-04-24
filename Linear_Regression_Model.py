# Polinomial Regression Model
# Can make Linear, Quadratic, Cubit, etc lines

# Initial weights (Define the curve type)
W = [0.1,0.2,0.1]
# Learning Rate
a = 0.0001
# Stopping criteria
E = 0.001

# Data
data = [[1,1],[2,4],[3,9],[4,16],[5,25],[6,36],[7,49],[8,64],[9,81],[10,100]]

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
            Grad[i] = Grad[i] + ((data[j][1] - Guess[j])*data[j][0]**i)/len(data)**i
    print(Grad)

    for i in range (0,len(W),1):
        W[i] = W[i] + a*Grad[i]
    return [W, Cost]

Cost = 1
count = 0
while Cost >= E:
# for i in range(0,10,1):
    [W, Cost] = GradientDescent(W)
    count = count +1
    if count >= 100000:
        break
print(W)
print(count)







