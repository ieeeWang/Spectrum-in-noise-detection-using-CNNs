# Spectrum-in-noise-detection-using-CNNs
This small project aims to detect a spectrum in noise and to demostrate it by using a simulated dataset with 10k samples.

To implement a spectum (in noise) detector, the file 'spectr_detect_train_v2.1.py' uses low-level APIs (e.g., tf.GradientTape) and cosomized 'loss' and 'metric' functions.
Whereas, the file 'spectr_detect_train_v2.2.py' uses high-level APIs (e.g., keras.model.fit) to implement similar functions.

Demos:

<img src="images\demo_10_testsamples.png" width="200px" height="800px" /> <img src="images\demo_10_testsamples_groundtruth.png" width="200px" height="800px" /> <img src="images\demo_10_testsamples_prediction.png" width="200px" height="800px" />


src="images\demo_ROC_PR.png" width="400px" height="400px" />
