import matplotlib.pyplot as plt
pra_file_path='/Users/pengqianhan/Library/CloudStorage/OneDrive-TheUniversityofAuckland/github-code/STfusion/output/trained_models_1129/log_test.txt'
out_rsme=[]
with open(pra_file_path, 'r') as reader:
		# print(train_file_path)
		for x in reader.readlines():
			if x.find('Test_Epoch')!=-1:
				temp=x.strip().split(' ')
				out_rsme.append(float(temp[-1]))
		optim_rsme=min(out_rsme)
		print('min rsme={},at {} epoch'.format(optim_rsme,out_rsme.index(optim_rsme)))

# plt.plot(train_loss_list, label = "Train loss")
plt.plot(out_rsme, label = "Validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()
