`Facial Emotion Recognition` on `fer2013` dataset using `TensorFlow`! (Accuracy ~ 65%)

DATA SET:
---------
- Download Data Set: `fer2013.bin` (63M) and `test_batch.bin` (7.9M) from https://goo.gl/ffmy2h

  Image Properties: `Size of an image` - 48 x 48 pixels (2304 bytes), `Size of a label` - number in (0..6) (1 byte) (0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Disgust, 5=Surprise, 6=Neutral).

  Data Set Format: `1st byte` is the label number and the `next 2304 bytes` are the image pixels.

- Create a data directory in your system: `/tmp/fer2013_data/`

- Put the training data set (28,709 images) in: `/tmp/fer2013_data/fer2013-batches-bin/fer2013.bin`

- Put the testing data set (3,589 images) in: `/tmp/fer2013_data/fer2013-batches-bin/test_batch.bin`

HOW TO TRAIN:
-------------
- Install `TensorFlow`: https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation
- Run `python fer2013_train.py`
- Run `python fer2013_eval.py` on fer2013.bin data (Training Precision)
- Run `python fer2013_eval.py` on test_batch.bin data (Evaluation Precision)

HOW TO DEMO:
------------
- From https://goo.gl/ffmy2h download the checkpoint files `checkpoint`, `model.ckpt-6000`, `model.ckpt-6000.meta` located in `65acc-checkpoint` dir.

- Copy these files into `/tmp/fer2013_train/`

- Inside the demo folder of the emotion-recognition project, run `./demo.sh IMG#`. Provide an `IMG#`, which is the row number in the `private-test-150.csv`, where each row corresponds to an image. There are 150 such rows/images.

- Executing `./demo.sh` outputs the `label` predicted by the trained model. This can be cross checked with the first value in the row of the csv file.

- To actually view the image and visually cross check the emotion, run `uint8-to-image.py` script on `private-test-150.csv`. This generates 150 .png image files with appropriate IMG# in the image file name.

STATS DASHBOARD:
----------------
- Run `tensorboard --logdir "/tmp"`
- Go to `http://0.0.0.0:6006/`
- This displays `events`, `images`, `graphs` and `histograms` for the train and eval runs on the model.

SCREENSHOTS:
------------
- https://goo.gl/rJfjYL

REFERENCES:
-----------
- Code references and examples from https://www.tensorflow.org
- Data Set from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
