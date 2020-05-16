
# Introduction
Hi there, in this post we are going to deploy the model we build in the [Image Classification] the development post. If you haven't gone through it yet please do has the goal for this site is to teach you how to build and deploy models seamlessly by going from development to production. We are going to classify images from a video feed and I am so exited to show you how to do this, let's jump right in.

# Tutorial Overview
1. Setup dependencies and Assets
2. Build Camera View
3. Handle Permissions
4. Build Camera Activity
5. Build Classification Activity
6. Running the app.


# 1. Setup Dependencies and Assets
From the previous development post, we saved two files i.e *labels.txt* and *tflite_model.tflite*, the former is our image classes and and latter the model. Open Android Studio and create a new empty project then in the project main, create an folder called assets and copy this two files there also rename MainActivity to ClassificationActivity.

The dependencies needed for this project is TensorFlow Lite and TensorFlow support, the former handles loading the model and running inference and latter for tensor operations, and CameraX to get camera input. Set noCompress for tflite. Open build.gradle(Module:app) and add the following.

```
// set no compress for tflite

compileOptions {
    sourceCompatibility JavaVersion.VERSION_1_8
    targetCompatibility JavaVersion.VERSION_1_8
}

android {
    aaptOptions {
        noCompress "tflite"
    }
}

dependencies {
    // TensorFlow Lite dependency
    implementation "org.tensorflow:tensorflow-lite:2.1.0"
    // Support library with image processing 
    implementation('org.tensorflow:tensorflow-lite-support:0.0.0-nightly') { changing = true }

    // Configure CameraX
    def camerax_version = '1.0.0-alpha06'
    implementation "androidx.camera:camera-core:${camerax_version}"
    implementation "androidx.camera:camera-camera2:${camerax_version}"
}


```

# 2. Build Camera View
We are going to utilize texture view used to display a live video stream and text view all this inside a FrameLayout.

# 3. Handle Permissions
Due to privacy issues, there is a need to request user permission to sensitive functions such as camera. Open android manifest and add the following

```
<uses-permission android:name="android.permission.CAMERA" />
```

Then handle the camera permissions in ClassificationActivity such that when the user launches the app we request for permission to access it, and remember it for the future. Add this to ClassificationActivity.

<script src="https://gist.github.com/kongkip/649c038fdad42d2b7fe08a5292ed97b4.js"></script>

# 4. Build Camera Activity
We will great function that creates and displays the camera preview, set transforms to compensate for changes in device orientation i.e calculating the center of TextureView, correcting the preview output and applying transformations to TextureView.

Add this to ClassificationActivity

<script src="https://gist.github.com/kongkip/a684808747141fd5d86c035e359329eb.js"></script>


# 5. Build Classification Activity
So far we have captured image feed from the camera and in this section we will process the it. 
CameraX has ImageAnalysis.Analyzer which enables image analysis and will get the image and since some devices rotates images on their own during capturing, we rotate it to its original position. We will get bitmap from the texture view.

In classification create a inner class to get the image

<script src="https://gist.github.com/kongkip/0cfb858f2b0afb49cbd9203dfb95d9b1.js"></script>

Let's now deal with ML logic, i.e load the image labels and the model, get the bitmap and resize it to 224 by 224 since our model requires this size.  We will set the mean to 127.0f and 128.0f standard deviation to to perform normalization during preprocessing ,probability mean and probability standard deviation to 0.0f and 1.0f respectively  to bypass normalization during postprocessing.

After running inference results will be collected and return the top-k probability, for our case we will return maximum of 3 results.

Create a new ImageClassifier Kotlin file class.

<script src="https://gist.github.com/kongkip/d8f8c98135543489c37c3da14d84c03e.js"></script>

Wait, what? that's it?. Yeah I know thats a lot to take in. Let me explain what the functions does
- *loadImage* - loads the bitmap and performs transformations on it, i.e cropping, resizing to required shape 224*225, rotation and finally preprocessing it by bypassing the normalization.

- *Result* - when running inference, the interpreter returns probabilities while labelling it, this class holds this results.

- *getTopKProbability* - the function ques the probabilities and utilizes comparator to compare confidence in the ques and later returns three results with higher probabilities.

- *classifyImage* -  takes input image buffer and output probability buffer runs inference on it to update output probability. It the labels it and returns top 3 results from the labelled probability.


Much of the processing work load is handled by TensorFlow support which has operations like normalization, image processing and more. We used it's FileUtil to load model and labels from the assets. ImageProcessor to process image buffer, TensorOperator to perform operations with NormalizeOp, TensorBuffer to create output probability buffer. Comments in the above code states how library functionalities where applied.

Now Run the app and point the camera to any furniture.


# Conclusion
Congratulations on reaching this far, I know it is a long tutorial but its worth the reading. It shows how to apply machine learning on a mobile device and clasping this concepts with make you confident use apply AI in your apps.
Please feel free to leave comments in the comment section below and mind being polite, we are here to learn. Cheers!!

# Refference
1. [Getting started with CameraX](https://www.raywenderlich.com/6748203-camerax-getting-started)

2. [TensorFlow Lite examples](https://github.com/tensorflow/examples)