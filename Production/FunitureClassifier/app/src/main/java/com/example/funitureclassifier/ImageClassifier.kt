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
    private var labels: List<String>? = null
    private var preprocessNormalizeOp : TensorOperator? = null
    private var postProcessNormalizeOp : TensorOperator? = null
    private var inputImageBuffer : TensorImage
    private lateinit var outputProbabilityBuffer : TensorBuffer
    private lateinit var probabilityProcessor : TensorProcessor
    private var probabilityShape : IntArray
    private var probabilityDataType : DataType


    init {
        val model = FileUtil.loadMappedFile(activity, MODEL_FILENAME)

        try {
            interpreter = Interpreter(model)
        }catch (e: Exception) {
            throw RuntimeException(e)
        }

        labels = FileUtil.loadLabels(activity, LABEL_FILENAME)
        // load model from assets
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
        // Creates the input tensor.

        postProcessNormalizeOp = NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)
        inputImageBuffer = TensorImage(imageDataType)
    }

    fun classifyImage(bitmap:Bitmap, sensorOrientation: Int):
            List<Result>{

        inputImageBuffer = loadImage(bitmap, sensorOrientation)
        // Creates the output tensor and its processor.
        outputProbabilityBuffer =
            TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
        probabilityProcessor = TensorProcessor.Builder().add(postProcessNormalizeOp).build()
        interpreter!!.run(inputImageBuffer.buffer, outputProbabilityBuffer.buffer.rewind())
        val labeledProbability =
            TensorLabel(labels!!, probabilityProcessor.process(outputProbabilityBuffer))
                .mapWithFloatValue

        return getTopKProbability(labeledProbability)
    }


    // Loads bitmap into a TensorImage.
    private fun loadImage(bitmap: Bitmap, sensorOrientation: Int): TensorImage {
        inputImageBuffer.load(bitmap)
        // Creates processor for the TensorImage.
        preprocessNormalizeOp = NormalizeOp(IMAGE_MEAN, IMAGE_STD)
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
//            if (id != null) resultString += "[$id] "
            if (title != null) resultString += title + " "
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
    const val IMAGE_MEAN = 127.0f
    const val IMAGE_STD = 128.0f
    const val PROBABILITY_MEAN = 0.0F
    const val PROBABILITY_STD = 1.0f
}