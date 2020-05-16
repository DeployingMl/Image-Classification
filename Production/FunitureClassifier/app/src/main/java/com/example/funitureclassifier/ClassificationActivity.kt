package com.example.funitureclassifier

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors
import kotlin.text.*

class ClassificationActivity : AppCompatActivity(), LifecycleOwner {

    private val REQUEST_CAMERA_CODE = 13
    private val PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    private val executor = Executors.newSingleThreadExecutor()

    private var textResult1: TextView? = null
    private var textResult2: TextView? = null
    private var textResult3: TextView? = null

    private lateinit var viewFinder: TextureView

    private var imageClassifier:ImageClassifier? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        viewFinder = findViewById(R.id.view_finder)
        imageClassifier = ImageClassifier(this)

        if (allPermissionsGranted()) {
            viewFinder.post { startCamera() }

        } else {
            ActivityCompat.requestPermissions(
                this, PERMISSIONS, REQUEST_CAMERA_CODE
            )
        }

        textResult1 = findViewById(R.id.text_result1)
        textResult2 = findViewById(R.id.text_result2)
        textResult3 = findViewById(R.id.text_result3)
    }

    private fun startCamera() {
        CameraX.unbindAll()

        val previewConfig = PreviewConfig.Builder().apply {
            setTargetResolution(Size(1920, 1080))
            setTargetRotation(viewFinder.display.rotation)
        }.build()

        // Build the viewFinder use case
        val preview = Preview(previewConfig)
        // Recompute layout every time viewFinder is updated
        preview.setOnPreviewOutputUpdateListener {
            val parent = viewFinder.parent as ViewGroup
            parent.removeView(viewFinder)
            parent.addView(viewFinder, 0)

            viewFinder.surfaceTexture = it.surfaceTexture
            updateTransform()

        }

        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            setImageReaderMode(
                ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE
            )
        }.build()

        // perform image analysis
        val imageAnalysis = ImageAnalysis(analyzerConfig).apply {
            setAnalyzer(executor, Classify())
        }

        CameraX.bindToLifecycle(this, preview, imageAnalysis)
    }

    private fun updateTransform() {
        val matrix = Matrix()

        // calculate center of texture view
        val centerX = viewFinder.width / 2f
        val centerY = viewFinder.height / 2f

        // correct the preview output
        val rotationDegrees = when (viewFinder.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }

        // apply transformations
        matrix.postRotate(-rotationDegrees.toFloat(), centerX, centerY)
        viewFinder.setTransform(matrix)
    }

    inner class Classify: ImageAnalysis.Analyzer {
        @SuppressLint("SetTextI18n")
        override fun analyze(imageProxy: ImageProxy?, rotationDegrees: Int) {
            // get bitmap from the camera finder view
            val imgBitmap = rotateImage(viewFinder.bitmap,
                rotationDegrees.toFloat())
            val recognitions = imageClassifier!!.classifyImage(
                imgBitmap, rotationDegrees)
            runOnUiThread {
                val empty = "       "
                textResult1!!.text = recognitions[0].title + empty +
                        String.format("%.2f%%", recognitions[0].confidence!! * 100.0f)

                textResult2!!.text = recognitions[1].title + empty +
                        String.format("%.2f%%", recognitions[1].confidence!! * 100.0f)

                textResult3!!.text = recognitions[2].title + empty+
                        String.format("%.2f%%", recognitions[2].confidence!! * 100.0f)

            }
        }
    }

    // rotate image to its original position
    private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    // Process result from permission request dialog and if request has been granted start camera
    // else show permission not granted
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CAMERA_CODE && grantResults.isNotEmpty() && grantResults[0] ==
            PackageManager.PERMISSION_GRANTED
        ) {
            if (allPermissionsGranted()) {
                viewFinder.post { startCamera() }
            } else {
                Toast.makeText(this, "Permissions not granted", Toast.LENGTH_LONG)
                    .show()
                finish()
            }
        }
    }

    // check if camera permission specified in the manifest has been granted
    private fun allPermissionsGranted() = PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    // Close the interpreter to avoid memory leakage
    override fun onDestroy() {
        super.onDestroy()
        imageClassifier!!.close()
    }

}
