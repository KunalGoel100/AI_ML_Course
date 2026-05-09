import math
a = 0.01
E = 0.0001
W_old = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
data = [[2,2,4],[3,5,9],[1,2,4],[4,4,8],[9,7,1],[3,5,8]]
data_y = [[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]]

# W_old = [[0,0,0],[0,0,0]]
# data = [[2,2],[3,5],[4,4],[9,7],[2,4],[2,2]]
# data_y = [[1,0],[0,1],[1,0],[0,1],[1,0],[1,0]]

data_x = [[1] + x for x in data]
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


    Y_Hat_temp1 = [0]*len(data_y)
    for j in range(0,len(data),1):
        Y_Hat_temp2 = [0] * len(data_y[0])
        for i in range(0,len(data[0]),1):
            Y_Hat_temp2[i] = math.exp(Z[j][i])
        Y_Hat_temp1[j] = Y_Hat_temp2
    # print(f"Y_Hat_temp1: {Y_Hat_temp1}")

    for j in range(0,len(data),1):
        temp = sum(Y_Hat_temp1[j])
        for i in range(0,len(data[0]),1):
            Y_Hat_temp1[j][i] /= temp
    Y_Hat = Y_Hat_temp1
    print(f"Y_Hat: {Y_Hat}")
    #
    Cost = 0
    for j in range(0,len(data),1):
        for i in range(0,len(data[0]),1):
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

    for j in range(0,len(data[0]),1):
        for i in range(0,len(W[0]),1):
            W[j][i] -= a*Slope[j][i]

    print(f"W: {W}")
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
    if Count >= 50000:
        break

##############################
## Testing
In = [0]*3

for i in range(0,3,1):
    In[i] = int(input("enter[a,b,c]"))
P = [0]*3
Z = [0]*3
for i in range(0,3,1):
        P[i] = W_old[i][0] + W_old[i][1]*In[0] + W_old[i][2]*In[1] + W_old[i][3]*In[2]
        Z[i] = math.exp(P[i])

s = sum(Z)
for i in range(0,3,1):
    Z[i] /= s

print(Z)
