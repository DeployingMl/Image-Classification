package com.example.funitureclassifier

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.RectF
import com.example.funitureclassifier.Keys.IMAGE_MEAN
import com.example.funitureclassifier.Keys.IMAGE_STD
import com.example.funitureclassifier.Keys.LABEL_FILENAME

import com.example.funitureclassifier.Keys.MAX_RESULTS
import com.example.funitureclassifier.Keys.MODEL_FILENAME
import com.example.funitureclassifier.Keys.PROBABILITY_MEAN
import com.example.funitureclassifier.Keys.PROBABILITY_STD
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.lang.Exception
import java.lang.RuntimeException
import java.util.*
import kotlin.collections.ArrayList


class ImageClassifier constructor(activity: Activity) {
    private var imageSizeX:Int? = null
    private var imageSizeY:Int? = null
    private var interpreter: Interpreter? = null
    private val options = Interpreter.Options()
    private var labels: List<String>? = null
    private var preprocessNormalizeOp : TensorOperator? = null
    private var postProcessNormalizeOp : TensorOperator? = null
    private var inputImageBuffer : TensorImage
    private lateinit var outputProbabilityBuffer : TensorBuffer
    private lateinit var probabilityProcessor : TensorProcessor
    private var probabilityShape : IntArray
    private var probabilityDataType : DataType


    init {
        // get model from the assets
        val model = FileUtil.loadMappedFile(activity, MODEL_FILENAME)

        // load the model interpreter
        try {
            interpreter = Interpreter(model, options)
        }catch (e: Exception) {
            throw RuntimeException(e)
        }

        // load labels from the disk
        labels = FileUtil.loadLabels(activity, LABEL_FILENAME)
        // Get input sizes
        val imageTensorIndex = 0
        val imageShape = interpreter!!.getInputTensor(imageTensorIndex).shape()
        imageSizeY = imageShape[1]
        imageSizeX = imageShape[2]

        val imageDataType =
            interpreter!!.getInputTensor(imageTensorIndex).dataType()
        val probabilityTensorIndex = 0
        probabilityShape =
            interpreter!!.getOutputTensor(probabilityTensorIndex).shape()
        probabilityDataType =
            interpreter!!.getOutputTensor(probabilityTensorIndex).dataType()

        // set mean to 0.0f and std 0.0f to pybass normalization
        preprocessNormalizeOp = NormalizeOp(IMAGE_MEAN, IMAGE_STD)
        
        // operation to de-quantize output probability
        postProcessNormalizeOp = NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)

        // creates an input tensor
        inputImageBuffer = TensorImage(imageDataType)
    }

    // classify image, running inference and returning labelled probabilities
    fun classifyImage(bitmap:Bitmap, sensorOrientation: Int):
            List<Result>{

        
        inputImageBuffer = loadImage(bitmap, sensorOrientation)

        // Creates the output tensor and its processor.
        outputProbabilityBuffer =
            TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
        
        // preprocessor of output probability
        probabilityProcessor = TensorProcessor.Builder().add(postProcessNormalizeOp).build()
        
        // run inference
        interpreter!!.run(inputImageBuffer.buffer, outputProbabilityBuffer.buffer.rewind())

        val labeledProbability =
            TensorLabel(labels!!, probabilityProcessor.process(outputProbabilityBuffer))
                .mapWithFloatValue

        return getTopKProbability(labeledProbability)
    }


    // Loads bitmap into a TensorImage and applies preprocessing 
    private fun loadImage(bitmap: Bitmap, sensorOrientation: Int): TensorImage {
        inputImageBuffer.load(bitmap)
        // Creates processor for the TensorImage.
        val cropSize = bitmap.width.coerceAtMost(bitmap.height)
        val numRotation = sensorOrientation / 90
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(imageSizeX!!, imageSizeY!!, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(Rot90Op(numRotation))
            .add(preprocessNormalizeOp)
            .build()
        return imageProcessor.process(inputImageBuffer)
    }

    // return top-k probability
    private fun getTopKProbability(labelProb: Map<String, Float>): List<Result> {
        val pq = PriorityQueue<Result>(
            MAX_RESULTS,
            kotlin.Comparator<Result> { lhs, rhs ->
                (rhs.confidence!!).compareTo(lhs.confidence!!)
            }
        )
        for ((i, j) in labelProb) {
            pq.add(Result("" + i, i, j, null))
        }

        val recognitions = ArrayList<Result>()
        val recognizeSize = pq.size.coerceAtMost(MAX_RESULTS)
        for (i in 0 until recognizeSize) recognitions += pq.poll()
        return recognitions
    }

    inner class Result(val id: String?,
                       val title: String?,
                       val confidence: Float?,
                       private var location: RectF?) {
        override fun toString(): String {
            var resultString = ""
            if (id != null) resultString += "[$id] "
            if (title != null) resultString += "$title "
            if (confidence != null) resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            if (location != null) resultString += location!!.toString() + " "
            return resultString.trim { it <= ' ' }
        }
    }

    // Close the interpreter after closing the app
    fun close() {
        interpreter!!.close()
    }
}



object Keys {

    // image label path
    const val LABEL_FILENAME = "labels.txt"
    // model path
    const val MODEL_FILENAME = "tflite_model.tflite"
    const val MAX_RESULTS = 3

    // set mean to  127.0f and std 128.0f to de-quantize during preprocessing
    const val IMAGE_MEAN = 127.0f
    const val IMAGE_STD = 128.0f

    // set mean to 0.0f and std 255.0f bypass normalization
    const val PROBABILITY_MEAN = 0.0F
    const val PROBABILITY_STD = 1.0f
}