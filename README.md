# FV_slip_prediction
This is repository for slip prediction of vision-based tactile sensor FingerVision.

## Usage guide
git clone this repo down to a directory, then download [dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yzhangfr_connect_ust_hk/EaUN_EOlAWRJhcy2QTNAoNIB3R6K2DX8bnePJAWwWKdtvA?e=GruTKH)
and unzip it along side with this repository folder.

1. First check if the dataloader in convLSTM_dataset.py can run correctly by:
```sh
python convLSTM_dataset.py
```
This script defines a customized dataset for dataloader of pytorch.

2. IF previous step passes, you can start training the frame prediction network by running convLSTM_frame_pred.py and should be able to see the drop of train/test loss versus epochs, also by:
```sh
python convLSTM_frame_pred.py
```
Meanwhile, run the following script in the shell and you should see the change of accuracy (finally it should hit over 95% of accuracy). This network is in charge of slip detection with a given frames of determined time steps (default T=10).
```sh
python convLSTM_slip_detection_1layer.py
```

3. Finaly, run the slip prediction accuracy test by:
```sh
python convLSTM_pred_slip.py
```
and it would give out accuracy over 90% that depends on how many frames ahead it is iterating. This is a stacking of previous trained slip detection network and frame prediction network that features future slip prediction capability with high accuracy.

further test of generalization ability to different contact object with variant geometry, stiffness, and size, etc.


