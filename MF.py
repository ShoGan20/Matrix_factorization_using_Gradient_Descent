from numpy import *
import pandas as pd



def matrix(R, d):
        
        
    Rate = mat(R)
    a, b  = shape(R)
    alpha =0.0002 # Learning Rate
    lmda = 0.04 # drop learning rate by factor of lambda
    epsil = 0.001 # lowest loss value
    epoch = 10000 # Iterations
    print("Target Matrix: ", Rate)

    item_val = mat(random.random((a, d)))
    user_val = mat(random.random((b, d)))

    for ep in range(epoch):
        for y in range(a):
            for z in range(b):
                if Rate[y, z] > 0:
                    err = Rate[y,z] - (user_val[z] * item_val[y].T)[0, 0]

                    user_val[z] += alpha * (err * item_val[y] - lmda * user_val[z])
                    item_val[y] += alpha * (err * user_val[z] - lmda * item_val[y])

            
        loss_val = 0
        for t in range(a):
            for v in range(b):
                if Rate[t, v] > 0:
                    loss_val += (Rate[t, v] - (user_val[v] * item_val[t].T)[0, 0]) ** 2 \
                        + lmda * (linalg.norm(item_val[t], 2) ** 2 + linalg.norm(user_val[v], 2) ** 2)

            
        if loss_val < epsil :
            break

        if ep % 1000 == 0:
                
            print(" epoch: ", ep, "Loss val: ", loss_val)

    return user_val, item_val





if __name__ == "__main__":
    # I'm taking input1.csv as example matrix, you can change it to any other csv file
    df = pd.read_csv('input1.csv')
    R = df.to_numpy()
    # Below is sample small matrix that I tried the code as well with
    # R = [[5, 3, 0, 1, 1],
    #     [4, 0, 0, 1, 2],
    #     [1, 1, 0, 5, 3],
    #     [0, 1, 5, 4, 0]]
    user_new, item_new = matrix(R, 20)
    pred = item_new * user_new.T

    print("Predicted Matrix: \n", pred)

