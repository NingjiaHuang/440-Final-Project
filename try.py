
import matplotlib.pyplot as plt
 
# line 1 points
x = [i for i in range(2, 11)]
acc = [0.47, 0.45, 0.53, 0.435, 0.47, 0.495, 0.445, 0.49, 0.54]
old_acc = [0.485, 0.5150000000000001, 0.5349999999999999, 0.5900000000000001, 0.4900000000000000, 0.5999999999999999, 0.465, 0.41, 0.62]
auc = [0.511, 0.525, 0.542, 0.499, 0.505, 0.535, 0.513, 0.519, 0.551]
old_auc = [0.49729978354978355, 0.5314646464646465, 0.5318294205794206, 0.5620093795093795, 0.49282439782439785, 0.583532301032301, 0.45076701076701076, 0.4512809412809412, 0.6130790043290044]
# plotting the line 1 points
plt.plot(x, acc, label = "acc")
plt.plot(x, old_acc, label = "old_acc")
plt.plot(x, auc, label = "auc") 
plt.plot(x, old_auc, label = "ald_auc")
# naming the x axis
plt.xlabel('number of clusters')
# naming the y axis
plt.ylabel('metric score')
# giving a title to my graph
plt.title('CCE Algorithm on Elephant')
 
# show a legend on the plot
plt.legend()
 
# function to show the plot
plt.show()