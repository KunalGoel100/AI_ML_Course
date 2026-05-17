import math
import random
a = 0.01
E = 0.001
e = 10**(-10)
W_old = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]

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
# mix
for j in range(200,300):
    while True:
        data_t = [random.randint(1,20) for _ in range(3)]
        evens = sum(1 for x in data_t if x % 2 == 0)
        odds = 3 - evens

        if evens > 0 and odds > 0:   # ✅ true mix
            break

    data[j] = data_t
    data_y[j] = [0,0,1]

# print(data)
# print(data_y)



# W_old = [0]
for i in range(0,len(data_y[0]),1):
    for j in range(0,len(data[0])+1,1):
        W_old[i][j] = random.random()

# print(W_old)

# W_old = [[0,0,0],[0,0,0]]
# data = [[2,2],[3,5],[4,4],[9,7],[2,4],[2,2]]
# data_y = [[1,0],[0,1],[1,0],[0,1],[1,0],[1,0]]
data = [[xx%2 for xx in x] for x in data]
data_x = [[1] + x for x in data]
print(data)
def Multi(W):
    Z = [0]*len(data)
    for j in range(0,len(data_y),1):
        Z_temp = [0]*len(data_y[0])
        for i in range(0,len(data_y[0]),1):
            for k in range(0,len(data_x[0]),1):
                Z_temp[i] += W[i][k]*data_x[j][k]
                # Z_temp[i] = W[i][0] + W[i][1]*data[j][0] + W[i][2]*data[j][1] + W[i][3]*data[j][2]
        Z[j] = Z_temp
    print(f"Z: {Z}")


    Y_Hat_temp1 = [0]*len(data)
    for j in range(0,len(data_y),1):
        Y_Hat_temp2 = [0] * len(data_y[0])
        for i in range(0,len(data_y[0]),1):
            Y_Hat_temp2[i] = math.exp(Z[j][i])
        Y_Hat_temp1[j] = Y_Hat_temp2
    # print(f"Y_Hat_temp1: {Y_Hat_temp1}")

    for j in range(0,len(data),1):
        temp = sum(Y_Hat_temp1[j])
        for i in range(0,len(data_y[0]),1):
            Check = Y_Hat_temp1[j][i] / temp
            if Check <= e:
                Y_Hat_temp1[j][i] = e
            else:
                Y_Hat_temp1[j][i] /= temp
    Y_Hat = Y_Hat_temp1
    print(f"Y_Hat: {Y_Hat}")
    #
    Cost = 0
    for j in range(0,len(data),1):
        for i in range(0,len(data_y[0]),1):
            Cost -= data_y[j][i]*math.log(Y_Hat[j][i])

    print(f"Cost {Cost}")

    Slope = [0]*len(data_y[0])
    for j in range(0,len(data_y[0]),1):
        Slope_temp = [0] * len(W[0])
        for i in range(0,len(W[0]),1):
            for k in range(0,len(data_y),1):
                Slope_temp[i] += -(data_y[k][j] - Y_Hat[k][j])*data_x[k][i]
        Slope[j] = Slope_temp

    # print(f"Slope: {Slope}")

    for j in range(0,len(data_y[0]),1):
        for i in range(0,len(W[0]),1):
            W[j][i] -= a*Slope[j][i]

    # print(f"W: {W}")
    return [W,Cost]

# for i in range(0,1000,1):
Cost = 1
Count = 0
diff = 1
while diff >= E:
    Cost_old = Cost
    [W_old,Cost] = Multi(W_old)
    diff = abs(Cost_old - Cost)
    Count += 1
    print(Count)
    if Count >= 5000:
        break

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
    print(Z)
    # check = 0



