import math
import copy
import random
a = 0.1
E = 0.001
e = 1e-15

######################################################3
## Training Data
data = [0]*300
data_y = [0]*300
# Even
for j in range(0,100,1):
    data_t = []
    for i in range(0,3,1):
        data_t.append(random.randrange(0,20,2))
    data[j] = data_t
    data_y[j] = [1,0,0]
# Odd
for j in range(100,200,1):
    data_t = []
    for i in range(0,3,1):
        data_t.append(random.randrange(1,21,2))
    data[j] = data_t
    data_y[j] = [0,1,0]
# mix
for j in range(200,300):
    while True:
        data_t = [random.randint(1,20) for _ in range(3)]
        evens = sum(1 for x in data_t if x % 2 == 0)
        odds = 3 - evens

        if evens > 0 and odds > 0:
            break

    data[j] = data_t
    data_y[j] = [0,0,1]
data = [[xx%2 for xx in x] for x in data]
data_x = [[1] + x for x in data]
print(data)

##########################################################
## Initial Weights
# W_old = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
W_old = []
for i in range(0,len(data_y[0]),1):
    W_old_temp = [0]*len(data_x[0])
    for j in range(0,len(data_x[0]),1):
        W_old_temp[j] = random.random()
    W_old.append(W_old_temp)
#########################################################
## Back Propagation (Finite difference method)
def BackProp(W):
    h = 0.01
    Cost1 = Multi(W)
    W_new = copy.deepcopy(W)
    for i in range(0,len(W),1):
        for j in range(0,len(W[0]),1):
            W2 = copy.deepcopy(W)
            W2[i][j] += h
            Cost2 = Multi(W2)
            Slope = (Cost2-Cost1)/h
            W_new[i][j] -= a*Slope
    return [W_new,Cost1]

#########################################################
## MultiNomial Classification Function
def Multi(W):
    Z = [0]*len(data)
    for j in range(0,len(data_y),1):
        Z_temp = [0]*len(data_y[0])
        for i in range(0,len(data_y[0]),1):
            for k in range(0,len(data_x[0]),1):
                Z_temp[i] += W[i][k]*data_x[j][k]
                # Z_temp[i] = W[i][0] + W[i][1]*data[j][0] + W[i][2]*data[j][1] + W[i][3]*data[j][2]
        Z[j] = Z_temp
    # print(f"Z: {Z}")
    Y_Hat_temp1 = [e]*len(data)
    for j in range(0,len(data_y),1):
        s = 0
        Y_Hat_temp2 = [0] * len(data_y[0])
        for i in range(0,len(data_y[0]),1):
            Y_Hat_temp2[i] = math.exp(Z[j][i])
            s = sum(Y_Hat_temp2)
        Y_Hat_temp1[j] = [Y_Hat_temp2_e/s for Y_Hat_temp2_e in Y_Hat_temp2]
    # print(f"Y_Hat_temp1: {Y_Hat_temp1}")

    Y_Hat = Y_Hat_temp1
    # print(f"Y_Hat: {Y_Hat}")
    #
    Cost = 0
    for j in range(0,len(data),1):
        for i in range(0,len(data_y[0]),1):
            Cost -= data_y[j][i]*math.log(Y_Hat[j][i] + e)
    # print(f"Cost {Cost}")
    return Cost

# for i in range(0,1000,1):
Cost = 1
Count = 0
diff = 1
while diff >= E:
    Cost_old = Cost
    # Multi(W_old)
    [W_old,Cost] = BackProp(W_old)
    print(Cost)
    diff = abs(Cost_old - Cost)
    Count += 1
    # print(Count)
    if Count >= 10000:
        break
# print(W_old)
##############################
## Testing
for t in range(0,5,1):
    In = [0]*3
    #
    for i in range(0,3,1):
        In[i] = float(input("enter[a,b,c]"))%2
    P = [0]*3
    Z = [0]*3
    for i in range(0,3,1):
            P[i] = W_old[i][0] + W_old[i][1]*In[0] + W_old[i][2]*In[1] + W_old[i][3]*In[2]
            Z[i] = math.exp(P[i])

    s = sum(Z)
    check = [0]*3
    for i in range(0,3,1):
        Z[i] /= s
        if Z[i] <= 0.5:
            check[i] = 0
        else:
            check[i] = 1

    if check == [1,0,0]:
        print("Even")
    elif check == [0,1,0]:
        print("Odd")
    else:
        print("Mix")


