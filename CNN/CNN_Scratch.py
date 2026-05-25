import numpy as np
from mpmath import zeros


# print(F)
## Convolution layer Function (Sum of Products)
def Conv(X,F):
    nxx, nxy = X.shape
    nfx,nfy = F.shape

    Ox = nxx-nfx+1
    Oy = nxy-nfy+1
    Output = np.zeros((Ox, Oy))
    # print(Output)


    for i in range(0,Ox,1):
        for j in range(0,Oy,1):
            Output[i,j] = sum(sum(X[i:i+nfx,j:j+nfy]*F))
    # print(Output)
    return Output

def ReLU(Input):
    # print(np.maximum(0, Input))
    return np.maximum(0, Input)

def MaxPolling(Input):
    Poling = [2,2]
    Stride = 1
    Ox = int((Input.shape[0]-Poling[0])/Stride + 1)
    Oy = int((Input.shape[1]-Poling[1])/Stride + 1)
    Output = np.zeros((Ox,Oy))
    # print(Output)

    for i in range(0,int(Ox),1):
        for j in range(0,int(Oy),1):
            Output[i,j] = (Input[i:i+Poling[0],j:j+Poling[1]]).max()
            # print(Input[i:i+Poling[0],j:j+Poling[1]])
    return Output

def Flatten(Input):
    a = Input.reshape(-1,1)
    # print(a)
    return a


def Dense(Input,W,bias):
    Z = np.dot(W,Input) + bias
    return Z

def Softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / np.sum(expZ)

############################################
# Example grayscale image (5x5)
X = np.array([
    [0,0,1,0,0],
    [0,1,1,1,0],
    [1,1,1,1,1],
    [0,1,1,1,0],
    [0,0,1,0,0]
], dtype=float)

# Label (digit class)
y = 1  # say class 1

## Filter (Kernal)
# F = np.random.randn(3,3)
F = np.array([
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
])
############################################
conv_out = Conv(X,F)
print(conv_out)
relu_out = ReLU(conv_out)
print(relu_out)
poling_out = MaxPolling(relu_out)
print(poling_out)
flatten_out = Flatten(poling_out)
print(flatten_out)

W = np.random.randn(4,flatten_out.size)
print(W)
bias = np.zeros([4,1])
dense_out = Dense(flatten_out,W,bias)
print(dense_out)
softmax_out = Softmax(dense_out)
print(softmax_out)
