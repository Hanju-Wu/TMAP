# Datasets

Download the classification data from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html), extract any compressed files, and place them in `logistic regression/dataset/` using the names below. The experiment driver does not download data automatically.

| Data set | Local file name |
| --- | --- |
| rcv1.train | `rcv1_train.binary` |
| rcv1.test | `rcv1_test.binary` |
| news20 | `news20.binary` |
| real-sim | `real-sim` |
| a9a | `a9a` |
| ijcnn1 | `ijcnn1` |

Run `setup_paths` and `build_mex` from the repository root to prepare the `libsvmread` interface. See [README.md](README.md) for compilation and experiment commands.

The Gaussian reconstruction driver generates its data automatically; no external LASSO data files are required.
