
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
From the previous development post, we saved two files i.e *labels.txt* and *tflite_model.tflite*, the former is our image classes and and latter the model. Open Android Studio and create a new empty project then in the project main, create an folder called assets and copy this two files there.

The dependencies needed for this project is TensorFlow, will help us load the model and run inference and CameraX to get camera input. Set noCompress for tflite. Open build.gradle(Module:app) and add the following.

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

    // Configure CameraX
    def camerax_version = '1.0.0-alpha06'
    implementation "androidx.camera:camera-core:${camerax_version}"
    implementation "androidx.camera:camera-camera2:${camerax_version}"
}


```

# 2. Build Camera View
We are going to utilize texture view used to display a live video stream and text view all this inside a FrameLayout.

# 3. Handle Permissions
Due to privacy issues, there is a need to request user permission to sensitive functions such as camera. Open android manifest and type the following

```
<uses-permission android:name="android.permission.CAMERA" />
```

Then handle the camera permissions in MainActivity such that when the user launches the app we request for permission to access it, and remember it for the future. Add this to MainActivity.

<!-- <script src="https://gist.github.com/kongkip/d12c6f00284bc86d89c5bf71d009250e.js"></script> -->


# 4. Build Camera Activity
We will great function that creates and displays the camera preview, set transforms to compensate for changes in device orientation i.e calculating the center of TextureView, correcting the preview output and applying transformations to TextureView.

Add this to MainActivity

<!-- <script src="https://gist.github.com/kongkip/9486cbe7cf8b793c85727fe8b764e398.js"></script> -->


# 5. Build Classification Activity
So far we have captured image feed from the camera and in this section we will process the it. 
CameraX has ImageAnalysis.Analyzer which enables image analysis and will get the image and since some devices rotates images on their own during capturing, we rotate it to its original position and finally convert it to bitmap.

Create a new Kotlin file class and add the following
<!-- <script src="https://gist.github.com/kongkip/695159258864df8ecb861d869bdf5077.js"></script> -->

Let's now deal with ML logic, i.e load the image labels and the model, get the bitmap and resize it to 224 by 224 since our model requires this size. We will also reshape the input.



