# Transfer Classifier




An experiment on multi-class classification vs. ensemble of class-specific classifiers using pretrained features.
 

 ## Repository Structure

 data/  : stores downloaded data
 model/  : store model files/checkpoints
 output/ 
 trainer/
    train_s.py    : train and test HGNet on Cifar10
    analyze_svm.py   : analyze pretrained features with different classifiers
 util/
    utils.py    : utility functions


## Documentations

### trainer/train_s.py

Train HGNet from scratch and test the model
```
    srun --gres=gpu:1 -w node05 python trainer/train_s.py --test_only false
```

Test HGNet using saved parameters 
```
    srun --gres=gpu:1 -w node05 python trainer/train_s.py --test_only true
```

### analyze_svm.py

Compare SVM based classifier performance and visualize features
```
    python analyze_svm.py
```
