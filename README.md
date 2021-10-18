Shonit Gangoly 
[@ShoGan20](https://github.com/ShoGan20)   
 
  
> Requirements (I have been using these versions but you can use any):   
    `numpy 1.21.2`<br> `pandas 1.3.4` <br><br>
    
## Topic

The aim is to perform Matrix Factorization using Gradient Descent.

## Function

1) So the code inputs the User Input Matrix and number of latent features the matrix should have.
2) It then divides the input matrix into User_values and Item_values
3) The function also has hyper parameters - Alpha(Learning Rate), lambda(dropping of learning rate throughout training), epsil(Lowest Loss value), Epochs(Number of iterations the code will run)
4) It then updates user and item matrice using Gradient Descent Matrix Factorization formula
5) Calculates loss using normalization of the values
6) Returns two matrices user, item
7) The predicted matrix is calculated by: user * transpose(item)

> To Run:  
    
    1. Unzip the file name 'Shonit_assignment1.zip'
    2. Open the file name 'MF.py'
    3. run the command 'python MF.py' in the terminal
    4. If you wish to change input values to the function, just change 'input1.csv' to any other file you would like. Just make sure the csv file is in the same directory as the python file.
 
> Output:
 1) Input Matrix
 2) Loss values at each iterative epoch
 3) Predicted Matrix
