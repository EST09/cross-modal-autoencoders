import matplotlib
import matplotlib.pyplot as plt

with open("/home/thomas/github/cross-modal-autoencoders/save_dir/train_ae_log.txt") as f:
    lines = f.readlines()
    data = lines[1:]

test_loss = []
epoch = []
train_loss = []

for txt in data:
    if txt.split()[0] == "Test":
        test_loss.append(float(txt.split()[3]))
    elif txt.split()[0] == "Epoch:":
        epoch.append(float(txt.split()[1]))
        train_loss.append(float(txt.split()[4]))

fig, axs = plt.subplots(2)
axs[0].plot(epoch, test_loss[1:])
axs[0].set_ylabel("test set loss")
axs[1].plot(epoch, train_loss)
axs[1].set_ylabel("train set loss")
plt.xlabel("epoch")
plt.show()


