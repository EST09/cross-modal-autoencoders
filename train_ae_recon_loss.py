import matplotlib
import matplotlib.pyplot as plt

with open("/home/thomas/github/cross-modal-autoencoders/save_dir/train_ae_log.txt") as f:
    lines = f.readlines()
    data = lines[1:]

test_loss = []
epoch = []

for txt in data:
    if txt.split()[0] == "Test":
        test_loss.append(float(txt.split()[3]))
    elif txt.split()[0] == "Epoch:":
        epoch.append(float(txt.split()[1]))

epoch.append(epoch[-1]+1)
plt.plot(epoch, test_loss)
plt.show()

