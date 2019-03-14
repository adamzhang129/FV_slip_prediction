# FV_slip_prediction
This is repository for slip prediction of vision-based tactile sensor FingerVision.

## Usage guide
First, git clone this repo down to a directory, then download [dataset]{https://hkustconnect-my.sharepoint.com/:u:/g/personal/yzhangfr_connect_ust_hk/EaUN_EOlAWRJhcy2QTNAoNIB3R6K2DX8bnePJAWwWKdtvA?e=GruTKH}
and unzip it along side with this repository folder.

First check if the dataloader in convLSTM_dataset.py can run correctly by:
```sh
python convLSTM_dataset.py
```

IF previous step passes, you can start training by running convLSTM_frame_pred.py and should be able to see the drop of train/test loss versus epochs, also by:
```sh
python convLSTM_frame_pred.py
```
Meanwhile, run the following in the shell and you should see the change of accuracy (finally it should hit over 95% of accuracy).
```sh
python convLSTM_slip_detection_1layer.py
```

Finaly, run the slip prediction accuracy test by:
```sh
python convLSTM_pred_slip.py
```
and it would give out accuracy over 90% that depends on how many frames ahead it is iterating.


