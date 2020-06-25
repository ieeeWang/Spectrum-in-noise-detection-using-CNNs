# Spectrum-in-noise-detection-using-CNNs
This project aims to detect a spectrum in noise and is demostrated using simulated dataset with 10k samples.

To do so, the file 'spectr_detect_train_v2.1.py' uses low-level APIs and cosomized 'loss' and 'metric' functions.
Whereas, the file 'spectr_detect_train_v2.2.py' uses high-level APIs (keras.model.fit) to implement similar functions.

Demos:

<img src="images\demo_10_testsamples.png" width="300px" height="800px" /> <img src="images\demo_10_testsamples_groundtruth.png" width="300px" height="800px" /> <img src="images\demo_10_testsamples_prediction.png" width="300px" height="800px" />
