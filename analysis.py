import numpy as np
import sklearn
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import ot 


# Use feature f only with SVM-based classifier

# Load precomputed features and labels from the NPC file
data_trn = np.load('data/feature_trn.npz')
data_tst = np.load( 'data/feature_tst.npz')
train_features = data_trn['f']
test_features= data_tst['f']
train_labels = data_trn['labels']
test_labels = data_tst['labels']

train_size = train_labels.shape[0]
test_size = test_labels.shape[0]
n_classes = 10
print('loaded',train_size,'training samples')
print('loaded',test_size,'testing samples')



# Visualize the classification results
""" 
plt.figure(figsize=(12, 6))
fs = [train_features, test_features]
ls = [train_labels,test_labels]
titles = ["train features","test features"]
for i in  range(2):
    # Perform t-SNE dimensionality reduction
    print('start tsne')
    tsne = TSNE(n_components=2, n_iter=300, random_state=42)
    reduced_features = tsne.fit_transform(fs[i])
    subsample_size = 400
    I = np.random.choice(len(reduced_features), size=subsample_size, 
                                      replace=False) 
    plt.title(titles[i])

    # Add subplot 
    plt.subplot(1,2,i+1)
    plt.scatter(reduced_features[I, 0], reduced_features[I, 1], 
                c=ls[i][I], cmap='viridis')
 
plt.suptitle("t-SNE Visualization of CIFAR-10 Classification Results (N=%d)"\
              % subsample_size )

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure 
print('save figure')
plt.savefig('output/tsne.pdf') 
"""

# 1) Train a 10-class SVM classifier
svm_classifier = svm.SVC()
svm_classifier.fit(train_features, train_labels)
svm_predictions = svm_classifier.predict(test_features)
svm_accuracy = accuracy_score(test_labels, svm_predictions)
print("SVM Classifier Accuracy:", svm_accuracy)

# 2) Train a one-vs-rest classifier
ovr_classifier = OneVsRestClassifier(svm.SVC())
ovr_classifier.fit(train_features, train_labels)
ovr_predictions = ovr_classifier.predict(test_features)
ovr_accuracy = accuracy_score(test_labels, ovr_predictions)
print("One-vs-Rest Classifier Accuracy:", ovr_accuracy)


# 3) Train 10 one-class SVM classifiers and aggregate the results
one_class_svms = []
one_class_predictions = np.zeros((test_size, n_classes)) #test_labels)

for class_label in range(n_classes):

    # Train a one-class SVM classifier
    one_class_svm = OneClassSVM()
    oc_train_features = train_features[train_labels==class_label,:] 
    one_class_svm.fit(oc_train_features)
    one_class_svms.append(one_class_svm)
    
    # Predict the class using the one-class SVM
    binary_predictions = one_class_svm.predict(test_features)
    print('Ratio of inliers in class%d: %f' % (class_label, 
            float(np.sum(np.max(binary_predictions,0)))/n_classes)
                                                )
    score_samples = one_class_svm.score_samples(test_features)
    one_class_predictions[:,class_label] = score_samples


acc = (np.argmax(one_class_predictions,axis=1)== test_labels ).sum()
one_class_accuracy = acc/test_size

print("One-Class SVM Classifier Accuracy:", one_class_accuracy)

#====================================================
# 4) Optimal transport + OneClassSVM (using both )

# apply optimal transport and apply barycentric mapping
def ot_semi_svm(Xs,Xt,ys,yt):
    oot_sinkhorn_semi = ot.da.SinkhornTransport(reg_e=1e-1)

    ot_sinkhorn_semi.fit(Xs=Xs, Xt=Xt, ys=ys, yt=yt)
    transp_Xs_sinkhorn_semi = ot_sinkhorn_semi.transform(Xs=Xs)
    return transp_Xs_sinkhorn_semi


 