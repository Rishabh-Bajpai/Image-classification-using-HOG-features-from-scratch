import functions as func
import numpy as np
import math
from sklearn import svm
from sklearn.svm import SVC # "Support Vector Classifier"
from sklearn.ensemble import RandomForestClassifier # Import Decision Tree Classifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.metrics import mean_absolute_error

filePath = '/root/Documents/CV/Ass2/2_class/'

total_img = 700
# total images = 700 of each class ( class = 1 is lake and class = 0 is plane)
training_img_no = math.floor(0.66*total_img)
#66% data is training data
print(training_img_no, "and", 2*training_img_no, " and", 2*(total_img-training_img_no), (total_img-training_img_no))
feature_matrix = np.empty([2*training_img_no, 8101])   #this is training dataset
#testing data
test_matrix = np.empty([2*(total_img-training_img_no), 8101])   #this is testing dataset
print('shapes are', feature_matrix.shape, test_matrix.shape)

for i in range(1, total_img + 1):
	img_no = f'{i:03}'
	filename=filePath + 'lake/lake_' + img_no + '.jpg' #file path
	feature_vect=func.hog(filename)
	feature_vect = np.append(feature_vect, 1)         # adding the label , 1 for lake image
	if (i <= training_img_no):
		print("training start:", i)
		feature_matrix[i-1] = feature_vect			#feature_matrix stores training hog features
	else:
		print("testing start:", i, "and", i- 1- training_img_no)
		test_matrix[i- 1- training_img_no] = feature_vect		#test_martix stores test hog features

print("test matrix" ,test_matrix.shape)

for i in range(1, total_img + 1):
	filename=filePath + 'airplane/airplane_' + img_no + '.jpg'
	feature_vect=func.hog(filename)
	feature_vect = np.append(feature_vect, 0)         # adding the label , 1 for airplane
	if (i <= training_img_no):
		print("training start:", i, "and", training_img_no + i-1 )
		feature_matrix[training_img_no + i-1] = feature_vect
	else:
		print("testing start:", i, "and", i - 1 - (total_img - training_img_no))
		test_matrix[i - 1 - (total_img - training_img_no)] = feature_vect
	


print(feature_matrix.shape)	
#splitting the data
train_X = feature_matrix[:, :8099]
train_X = np.nan_to_num(train_X)
train_Y = feature_matrix[:, 8100]
print(np.where(np.isnan(train_X)))

print(test_matrix.shape)
test_X = test_matrix[:, :8099]
test_Y = test_matrix[:, 8100]

#uncomment classifier
clf = svm.SVC()
# clf = RandomForestClassifier(n_estimators=200)

clf.fit(train_X, train_Y) #fit the model
pred_y = clf.predict(test_X)
#performance parameters
print("Accuracy:",accuracy_score(test_Y, pred_y))
print(classification_report(test_Y, pred_y))
print(mean_absolute_error(test_Y, pred_y))



# print(classification_report(train_Y, test_Y))

'''
imagePre=func.preprocess(filename)
gradiant,gradiant_angle=func.cal_gradiant(imagePre)
histogram_block_mat=func.hist_gradiant(gradiant,gradiant_angle)
histogram_block_mat_norm=func.Block_norm(histogram_block_mat)
feature_vect=np.reshape(histogram_block_mat_norm,(8100))
print(histogram_block_mat_norm.shape)

print(feature_vect)

'''
