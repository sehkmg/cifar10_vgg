import pandas as pd
import matplotlib.pyplot as plt

df_loss = pd.read_csv('loss.txt', header=None, sep='\t')
df_error = pd.read_csv('error.txt', header=None, sep='\t')

train_loss = df_loss[0].values
valid_loss = df_loss[1].values
test_loss = df_loss[2].values

train_error = df_error[0].values
valid_error = df_error[1].values
test_error = df_error[2].values

plt.figure()
plt.plot(train_loss, label='train')
plt.plot(valid_loss, label='valid')
plt.plot(test_loss, label='test')
plt.legend()
plt.savefig('loss.png', format='png', bbox_inches='tight')

plt.clf()
plt.plot(train_error, label='train')
plt.plot(valid_error, label='valid')
plt.plot(test_error, label='test')
plt.legend()
plt.savefig('error.png', format='png', bbox_inches='tight')
