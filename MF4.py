import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
import math
import random
#from sklearn.model_selection import train_test_split


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    test_index = np.zeros((ratings.shape[0], 10)).astype(int)
    for user in xrange(ratings.shape[0]):
        test_index[user] = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=False)
        train[user, test_index[user]] = 0.
        test[user, test_index[user]] = ratings[user, test_index[user]]
    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test, test_index

# Data preparation
def dataPrep():
    filePath = 'C:\\Users\Gaurav\PycharmProjects\MachineLearningHomework\Hw9\\'
    ratings_df = pd.read_csv(filePath+'training_rating.dat', sep="::", names=['UserID','MovieID','Rating'], engine='python')
    ratings_df= ratings_df.dropna(how='any')
    ratings_df = (ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0))
    #ratings_df_new=ratings_df.copy(deep=True)
    ratings = np.array(ratings_df.as_matrix())
    print 'ratings before',ratings
    ratings=ratings-3
    print 'ratings after',ratings
    #Xtrain, Xvalidation, val_index = train_test_split(ratings)
    # val_index = np.where(Xvalidation != 0)
    print 'ratings shape: ', ratings.shape
    # print 'type of ratings_df',type(ratings_df)
    #print 'Xvalidation shape: ', Xvalidation.shape
    #return Xtrain, Xvalidation, val_index

    # print 'row names of dataframe',ratings_df.index.values.tolist()
    # print 'column names of dataframe',ratings_df.columns.values.tolist()
    return ratings,ratings_df

def genUV(ratings,k,lamda):
    u = np.random.rand(ratings.shape[0],k)
    v = np.random.rand(ratings.shape[1],k)
    u_bias=np.zeros(shape=ratings.shape[0])
    print 'u_bias shape: ', u_bias.shape
    v_bias=np.zeros(shape=ratings.shape[1])
    print 'v_bias shape: ', v_bias.shape
    index = np.where(ratings != -3)
    #print 'index length',index.shape
    g_bias=np.mean(ratings[index])
    print 'g_bias ',g_bias
    #u = np.zeros(shape=(ratings.shape[0],10))       # m x k
    #v = np.zeros(shape=(ratings.shape[1],10))      # n x k
    epochs = 100
    lr = 0.002
    #lamda = 0.01
    u_bias_lamda=0.01
    v_bias_lamda=0.01
    #index = np.nonzero(ratings)
    print "u: ",u
    print "v: ",v

    flag = 0
    while flag<100:
        print 'flag: ',flag
        for x,y in zip(index[0],index[1]):
            e = ratings[x,y] - (g_bias+u_bias[x]+v_bias[y]+np.dot(u[x,:],v[y,:].T))
            u_bias[x] = u_bias[x]+ lr * (e - u_bias_lamda * u_bias[x])
            v_bias[y] = v_bias[y] + lr * (e - v_bias_lamda * v_bias[y])
            #print np.dot(u[x,:],v[y,:].T)
            #if flag==2:
                #print 'e value: ',e
                #print u[1:5,]
                #print v[1:5,]
            u[x, :] = u[x, :] - lr *(e * (-v[y, :])+ lamda * u[x, :])
            v[y, :] = v[y, :] - lr * (e * (-u[x, :]) + lamda * v[y, :])
        flag += 1
        # filePath = 'C:\Users\Gaurav\PycharmProjects\MachineLearningHomework\Hw9\\'

        # np.savetxt(filePath+'u.csv',u)
        # np.savetxt(filePath+'v.csv',v)
    return u,v,index,g_bias,u_bias,v_bias

def predict_ratings(u,v,g_bias,u_bias,v_bias):
    mean_center = 3
    pred_rating_matrix = np.dot(u, v.T)
    print 'before mean centering',pred_rating_matrix
    pred_rating_matrix=pred_rating_matrix+mean_center+g_bias
    print 'After mean centering',pred_rating_matrix
    for i in range(u.shape[0]):
        pred_rating_matrix[i]= pred_rating_matrix[i]+v_bias
    for j in range(v.shape[0]):
        pred_rating_matrix[:,j]=pred_rating_matrix[:,j]+u_bias
    print 'updated predicted matrix is',pred_rating_matrix
    # print 'prm shape',pred_rating_matrix.shape
    # for i, j in zip(index[0], index[1]):
    #     pred_rating_matrix[i,j]+=(g_bias+u_bias[i]+v_bias[j])
    # return pred_rating_matrix


    # print 'printing', u.shape,v.shape,u.shape[0],v.shape[0]
    # predictions = np.zeros((u.shape[0],v.shape[0]))
    # print 'shape of predictions: ', predictions.shape
    #
    # for user in range(u.shape[0]):
    #     for movie in range(v.shape[0]):
    #         total_bias=g_bias+u_bias[user-1]+v_bias[movie-1]+mean_center
    #         predictions[user-1, movie-1] = np.dot(u[user-1,:],v[movie-1,:])+total_bias
    return pred_rating_matrix


def calc_rmse(pred_ratings, XTrain, Xvalidation, val_index):
    train_error = 0.0
    for i in range(pred_ratings.shape[0]):
        train_index = [i for i in xrange(pred_ratings.shape[1]) if i not in val_index[0]]
        train_error = train_error + np.sum((pred_ratings[i, train_index] - XTrain[i, train_index]) ** 2)
    train_error = np.sqrt(train_error / (pred_ratings.shape[0] * (pred_ratings.shape[1] - 10)))
    test_error = 0.0
    for i in range(pred_ratings.shape[0]):
        test_error = test_error + np.sum((pred_ratings[i, val_index[i]] - Xvalidation[i, val_index[i]]) ** 2)
    test_error = np.sqrt(test_error / (pred_ratings.shape[0] * 10))
    return train_error, test_error


# def calc_rmse(pred_rating_matrix,true_ratings,val_index):
#     #true_ratings=np.genfromtxt("results.csv",delimiter=",")
#     error=0
#     for i, j in zip(val_index[0],val_index[1]):
#         #zip(index[0], index[1]):
#         error+=(true_ratings[i,j]-pred_rating_matrix[i,j])**2
#     print 'length: ',len(true_ratings[0])
#     print 'error:',error
#     error=math.sqrt(error/len(val_index[0]))
#     # for i,j in zip(index[0],index[1]):
#     #     error+=(true_ratings[i,j]-pred_rating_matrix[i,j])**2
#     # error=math.sqrt(error/len(index[0]))
#     return error
####################--------test predictions------###############_
def gen_test_instances(pred_ratings,test_data,extract_from_df):
    test_pred_values=[]
    #for i,j in zip(test_data[0],test_data[1]):
    df_rowlist=extract_from_df.index.values.tolist()
    df_col_list=extract_from_df.columns.values.tolist()
    pred_ratings_df= pd.DataFrame(pred_ratings,index=df_rowlist,columns=df_col_list)
    test_n_0_train=list(set(test_data[:,0])-set(df_col_list))
    test_n_1_train=list(set(test_data[:,1])-set(df_col_list))
    #movies which are not present assign random values else assign the predicted values
    for i in range(len(test_data)):
        if(test_data[i][1] in test_n_1_train):
            test_pred_values.append(random.randrange(1,5))
        else:
            test_pred_values.append(pred_ratings_df[test_data[i][1]][test_data[i][0]])
    #print 'rmse is: ',test_pred_values
    return test_pred_values


def main():

    ratings,ratings_df_new = dataPrep()
    k = [10]
    lamda = [0.01]
    print 'type of ratings_df', type(ratings_df_new)
    for i in k:
        for j in lamda:
            u,v,index,g_bias,u_bias,v_bias=genUV(ratings,i,j)

            pred_rating_matrix = predict_ratings(u, v, g_bias, u_bias, v_bias)

            filePath = 'C:\Users\Gaurav\PycharmProjects\MachineLearningHomework\Hw9\\'
            test_data=np.genfromtxt(os.path.join(filePath,"testing.dat"),delimiter=' ')
            print 'test data is', test_data
            test_pred_values=gen_test_instances(pred_rating_matrix,test_data,ratings_df_new)
            np.savetxt('predicted_y',test_pred_values,delimiter=',')
            #train_error, test_error = calc_rmse(pred_rating_matrix, XTrain, XValidation, val_index)
            #print 'k: ', i, '   lamda: ', j, '   Train_Error: ',train_error,'   Error: ', test_error

    #print 'Train error', trainerror,'Testerror: ',testerror
    #rmse=calc_rmse(pred_rating_matrix,true_ratings)

    # print 'error is',rmse
main()

# XTrain, Xvalidation, val_index = dataPrep()
# k = [10, 15, 20, 25]
# lamda = [0.01, 0.1, 1, 5, 10]
# for i in k:
#     for j in lamda:
#         u, v = genUV(XTrain, i, j)
#         pred_ratings = np.dot(u,v.T)
#         train_error, test_error = calc_rmse(pred_ratings, XTrain, Xvalidation, val_index)
#         print 'k: ', i, '   lamda: ', j, '   Train_Error: ',train_error,'   Error: ', test_error