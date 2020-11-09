# Spectrum-in-noise-detection-using-CNNs
This small project aims to detect a spectrum in noise and to demostrate it by using a simulated dataset with 10k samples.

To implement a spectum (in noise, Gaussian white noise with mean=255 and variance=60) detector, the file 'spectr_detect_train_v2.1.py' uses low-level APIs (e.g., tf.GradientTape) and cosomized 'loss' and 'metric' functions.
Whereas, the file 'spectr_detect_train_v2.2.py' uses high-level APIs (e.g., keras.model.fit) to implement similar functions.

Demos:

<img src="images\demo_10_testsamples.png" width="200px" height="800px" /> <img src="images\demo_10_testsamples_groundtruth.png" width="200px" height="800px" /> <img src="images\demo_10_testsamples_prediction.png" width="200px" height="800px" />


<img src ="images\demo_ROC_PR.png" width="800px" height="400px" />



## Google Colab training
To learn more about Google Colab Free gpu (Tesla T4 with the computing power 7!) training, visit this [tutorial](https://pylessons.com/YOLOv3-TF2-GoogleColab/)
and refer to my next project [ASSR_detector_CNN](https://github.com/ieeeWang/ASSR_detector_CNN).
