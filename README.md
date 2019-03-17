# Implementation of "Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data" (AFL)

### Paper

YapingZhang, Shan Liang, Shuai Nie, Wenju Liu, Shouye Peng, ["Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data"](https://www.sciencedirect.com/science/article/pii/S0167865518300412?via%3Dihub), PR Letter 2018

### Dependency

* Please use python3, as we cannot guarantee its compatibility with python2.
* The version of Tensorflow we use is 1.10.1.
* Other depencencies:

    ```
    pip install keras
    ```

### Usage
0. Clone the repo.

    ```shell
    git clone https://github.com/AprilYapingZhang/AFL.git
    cd AFL
    ```

#### Using ready-made data

1. Download the data from  [Baidu Yun](https://pan.baidu.com/s/15A6SL7JFIQaozUIEhtekxQ) with passwd `u8vz` , to the repo root, and uncompress it.

    ```shell
    tar -xf data.tar.gz
    ```

2. Make sure the structure looks like the following:

    ```shell
    data/:
    CASIA_HWDB_1.0_1.1_data
    data/CASIA_HWDB_1.0_1.1_data:
    norm_hand_pair_3755.hdf5  trn-HWDB1.0-1.1-3756-uint8.hdf5  tst-HWDB1.0-1.1-3756-uint8.hdf5

    ```

3. Run model
    * Run Baseline:

        ```shell
        python baseline.py --data_dir ./data
        ```

    * Run AFL model:

        ```shell
        python main.py --data_dir ./data
        ```


### Bibtex
```
@article{zhang2018robust,
  title={Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data},
  author={Zhang, Yaping and Liang, Shan and Nie, Shuai and Liu, Wenju and Peng, Shouye},
  journal={Pattern Recognition Letters},
  volume={106},
  pages={20--26},
  year={2018},
  publisher={Elsevier}
}
```
