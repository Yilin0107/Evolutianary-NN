import numpy as np
filea = np.load('./model/LinearCombination_NSGA2/step1/hypervolumes/fnn2_digits-00_Hypervolume.npz')
fileb = np.load('./model/LinearCombination_NSGA2/step1/hypervolumes/fnn2_digits-01_Hypervolume.npz')
hva = filea['hypervolume']
hvb = fileb['hypervolume']
print(hva)
print(hvb)
file1 = np.load('./model/LinearCombination_NSGA2/step2/hypervolumes/fnn2_digits-00_Hypervolume.npz')
file2 = np.load('./model/LinearCombination_NSGA2/step2/hypervolumes/fnn2_digits-01_Hypervolume.npz')
hv1 = file1['hypervolume']
hv2 = file2['hypervolume']
print(hv1)
print(hv2)

'''
file = np.load('./model/GLMutation_NSGA2/final_performance/fnn2_digits_GLMutation_NSGA2.npz')
pre = file['precision']
#print(pre)
rec = file['recall']
#print(rec)
ent = file['entropy']
#print(ent)
acc = file['accuracy']
print(acc)

file = np.load('./model/LinearCombination_NSGA2/step2/final_performance/fnn2_digits-00_LinearCombination_NSGA2.npz')
pre_ = file['precision']
print(pre_.shape)
rec_ = file['recall']
#print(rec_)
ent_ = file['entropy']
#print(ent_)
acc_ = file['accuracy']
#print(acc_)
'''
