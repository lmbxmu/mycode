**Noticeably**, the tools of data, util, vlfeat, affinity.m, evaluate.m and startup.m come from <a href="https://github.com/fcakir/mihash">here</a> and we highly **appreciate the authors' contributions**.

**The followings are our contributions:**

**MY_ONLINE_HASHING_METHOD** directory contains our online hashing methods including

(1) **HCOH.m**  ==> <a href="https://dl.acm.org/citation.cfm?id=3240519">Supervised Online Hashing via Hadamard Codebook Learning </a>, ACM MM 2018. 

(2) **BSODH.m**  ==> <a href ="https://arxiv.org/abs/1901.10185">Towards Optimal Discrete Online Hashing with Balanced Similarity </a>, AAAI 2019.

(3) **HMOH.m**  ==> <a href ="https://arxiv.org/abs/1905.04454">Hadamard Matrix Guided Online Hashing </a>, IJCV 2020 (Under Review).

**HMOH shows the best performance and is encouraged to compare.**

**Running Steps**:

First, run **./data/download_data.sh** to download the datasets of CIFAR10-VGG16-fc7, ImageNet-VGG, LabelMe-GIST and Places205_AlexNet_fc7_PCA128. Click <a href="https://drive.google.com/file/d/1iVOD4dsUSxujmN0F66w-JuEqljFZpaPK/view">here</a> to download the MNIST dataset. Click <a href="https://drive.google.com/file/d/1jJThCuiR2G0cQUfd0QZrKWnlLxy5IiIJ/view">here</a> to download the NUS-WIDE dataset.

Second, put all downloaded datasets into **./data** directory.

Third, run **startup.m**.

Fourth, run **our source codes**.
