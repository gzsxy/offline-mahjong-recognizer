package com.example.majiang.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.example.majiang.model.Detection
import com.example.majiang.model.TileClasses
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

private const val TAG = "TileDetector"

/**
 * YOLOv8 TFLite 检测器。
 *
 * Ultralytics 导出的模型通常输出 [1, 4+nc, 8400]，且框坐标经过
 * NormalizeCoords 变为 0..1。这里同时兼容 [1, N, C] 布局、不同类别数
 * 和 int8/uint8 张量，避免模型重新导出后 Android 端静默解析错误。
 */
class TileDetector(
    context: Context,
    assetName: String = "mahjong_28cls_int8.tflite",
    private val confThreshold: Float = 0.45f,
    private val normalizedOutputCoordinates: Boolean = true,
    private val preferGpu: Boolean = true,
    private val modelClassCount: Int = TileClasses.CLASS_COUNT,
    private val classIdMapper: (Int) -> Int = { it }
) : DetectionBackend {

    private enum class InputLayout { NCHW, NHWC }

    private data class InterpreterBundle(
        val interpreter: Interpreter,
        val gpuDelegate: GpuDelegate?
    )

    private val modelBuffer: ByteBuffer = loadModel(context, assetName)
    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    private val inputLayout: InputLayout
    private val inputWidth: Int
    private val inputHeight: Int
    private val inputDataType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val outputChannelFirst: Boolean
    private val outputChannels: Int
    private val outputCount: Int
    private val outputDataType: DataType
    private val outputScale: Float
    private val outputZeroPoint: Int
    private var closed = false

    init {
        require(confThreshold in 0f..1f) { "置信度阈值必须在 0 到 1 之间" }
        require(modelClassCount > 0) { "模型类别数必须大于 0" }

        val bundle = createInterpreter(modelBuffer, preferGpu)
        interpreter = bundle.interpreter
        gpuDelegate = bundle.gpuDelegate

        try {
            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape()
            require(inputShape.size == 4 && inputShape[0] == 1) {
                "不支持的模型输入形状：${inputShape.contentToString()}"
            }
            inputLayout = when {
                inputShape[1] == 3 -> InputLayout.NCHW
                inputShape[3] == 3 -> InputLayout.NHWC
                else -> error("模型输入必须包含 3 个颜色通道：${inputShape.contentToString()}")
            }
            inputHeight = if (inputLayout == InputLayout.NCHW) inputShape[2] else inputShape[1]
            inputWidth = if (inputLayout == InputLayout.NCHW) inputShape[3] else inputShape[2]
            require(inputWidth > 0 && inputHeight > 0) { "模型输入尺寸无效" }
            inputDataType = inputTensor.dataType()
            inputScale = inputTensor.quantizationParams().scale
            inputZeroPoint = inputTensor.quantizationParams().zeroPoint

            val outputTensor = interpreter.getOutputTensor(0)
            val outputShape = outputTensor.shape()
            require(outputShape.size == 3 && outputShape[0] == 1) {
                "不支持的模型输出形状：${outputShape.contentToString()}"
            }
            val expectedChannels = 4 + modelClassCount
            outputChannelFirst = when {
                outputShape[1] == expectedChannels -> true
                outputShape[2] == expectedChannels -> false
                else -> error(
                    "模型输出通道数不是 ${expectedChannels}：${outputShape.contentToString()}"
                )
            }
            outputChannels = if (outputChannelFirst) outputShape[1] else outputShape[2]
            outputCount = if (outputChannelFirst) outputShape[2] else outputShape[1]
            outputDataType = outputTensor.dataType()
            outputScale = outputTensor.quantizationParams().scale
            outputZeroPoint = outputTensor.quantizationParams().zeroPoint
        } catch (error: Throwable) {
            interpreter.close()
            gpuDelegate?.close()
            throw error
        }
    }

    /** 对一块切片做检测，返回切片坐标系下的结果（未做 NMS）。 */
    @Synchronized
    override fun detect(slice: Bitmap): List<Detection> {
        check(!closed) { "检测器已关闭" }
        require(!slice.isRecycled) { "不能检测已回收的 Bitmap" }

        val inputBitmap = if (slice.width == inputWidth && slice.height == inputHeight) {
            slice
        } else {
            Bitmap.createScaledBitmap(slice, inputWidth, inputHeight, true)
        }

        try {
            val inputTensor = interpreter.getInputTensor(0)
            val inputBuffer = ByteBuffer
                .allocateDirect(inputTensor.numBytes())
                .order(ByteOrder.nativeOrder())
            val pixels = IntArray(inputWidth * inputHeight)
            inputBitmap.getPixels(
                pixels,
                0,
                inputWidth,
                0,
                0,
                inputWidth,
                inputHeight
            )

            if (inputLayout == InputLayout.NCHW) {
                for (channel in 0..2) {
                    for (pixel in pixels) {
                        putInputValue(inputBuffer, redGreenBlue(pixel, channel) / 255f)
                    }
                }
            } else {
                for (pixel in pixels) {
                    for (channel in 0..2) {
                        putInputValue(inputBuffer, redGreenBlue(pixel, channel) / 255f)
                    }
                }
            }
            inputBuffer.rewind()

            val outputTensor = interpreter.getOutputTensor(0)
            val outputBuffer = ByteBuffer
                .allocateDirect(outputTensor.numBytes())
                .order(ByteOrder.nativeOrder())
            interpreter.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
            val values = readOutput(outputBuffer)

            fun value(channel: Int, index: Int): Float = if (outputChannelFirst) {
                values[channel * outputCount + index]
            } else {
                values[index * outputChannels + channel]
            }

            val scaleX = slice.width.toFloat() / inputWidth
            val scaleY = slice.height.toFloat() / inputHeight
            val out = ArrayList<Detection>(64)
            for (index in 0 until outputCount) {
                var bestId = -1
                var bestScore = 0f
                for (classId in 0 until modelClassCount) {
                    val score = value(4 + classId, index)
                    if (score > bestScore) {
                        bestScore = score
                        bestId = classId
                    }
                }
                if (bestId < 0 || !bestScore.isFinite() || bestScore < confThreshold) continue

                val rawCx = value(0, index)
                val rawCy = value(1, index)
                val rawW = value(2, index)
                val rawH = value(3, index)
                if (!listOf(rawCx, rawCy, rawW, rawH).all { it.isFinite() }) continue

                val cx = if (normalizedOutputCoordinates) rawCx * slice.width else rawCx * scaleX
                val cy = if (normalizedOutputCoordinates) rawCy * slice.height else rawCy * scaleY
                val width = if (normalizedOutputCoordinates) rawW * slice.width else rawW * scaleX
                val height = if (normalizedOutputCoordinates) rawH * slice.height else rawH * scaleY
                if (width <= 0f || height <= 0f) continue

                val left = (cx - width / 2f).coerceIn(0f, slice.width.toFloat())
                val top = (cy - height / 2f).coerceIn(0f, slice.height.toFloat())
                val right = (cx + width / 2f).coerceIn(0f, slice.width.toFloat())
                val bottom = (cy + height / 2f).coerceIn(0f, slice.height.toFloat())
                if (right <= left || bottom <= top) continue

                val mappedId = classIdMapper(bestId)
                if (mappedId !in 0 until TileClasses.CLASS_COUNT) continue
                out += Detection(RectF(left, top, right, bottom), mappedId, bestScore)
            }
            return out
        } finally {
            if (inputBitmap !== slice && !inputBitmap.isRecycled) inputBitmap.recycle()
        }
    }

    private fun putInputValue(buffer: ByteBuffer, normalized: Float) {
        when (inputDataType) {
            DataType.FLOAT32 -> buffer.putFloat(normalized)
            DataType.UINT8 -> buffer.put(
                quantize(normalized, inputScale, inputZeroPoint, 0, 255).toByte()
            )
            DataType.INT8 -> buffer.put(
                quantize(normalized, inputScale, inputZeroPoint, -128, 127).toByte()
            )
            else -> error("不支持的模型输入类型：$inputDataType")
        }
    }

    private fun readOutput(buffer: ByteBuffer): FloatArray {
        val values = FloatArray(outputChannels * outputCount)
        when (outputDataType) {
            DataType.FLOAT32 -> buffer.asFloatBuffer().get(values)
            DataType.UINT8 -> for (index in values.indices) {
                values[index] = dequantize(
                    buffer.get(index).toInt() and 0xFF,
                    outputScale,
                    outputZeroPoint
                )
            }
            DataType.INT8 -> for (index in values.indices) {
                values[index] = dequantize(
                    buffer.get(index).toInt(),
                    outputScale,
                    outputZeroPoint
                )
            }
            else -> error("不支持的模型输出类型：$outputDataType")
        }
        return values
    }

    private fun redGreenBlue(pixel: Int, channel: Int): Int = when (channel) {
        0 -> (pixel shr 16) and 0xFF
        1 -> (pixel shr 8) and 0xFF
        else -> pixel and 0xFF
    }

    private fun quantize(value: Float, scale: Float, zeroPoint: Int, min: Int, max: Int): Int {
        if (scale <= 0f) return zeroPoint.coerceIn(min, max)
        return (value / scale + zeroPoint).roundToInt().coerceIn(min, max)
    }

    private fun dequantize(value: Int, scale: Float, zeroPoint: Int): Float =
        if (scale > 0f) (value - zeroPoint) * scale else value.toFloat()

    private fun createInterpreter(model: ByteBuffer, preferGpu: Boolean): InterpreterBundle {
        if (preferGpu) {
            val delegate = runCatching { GpuDelegate() }
                .onFailure { Log.w(TAG, "GPU Delegate 创建失败，回退 CPU", it) }
                .getOrNull()
            if (delegate != null) {
                val gpuInterpreter = runCatching {
                    model.rewind()
                    Interpreter(model, Interpreter.Options().apply {
                        numThreads = 4
                        addDelegate(delegate)
                    })
                }.onFailure {
                    Log.w(TAG, "GPU delegate 不兼容当前模型，回退 CPU", it)
                }.getOrNull()
                if (gpuInterpreter != null) {
                    Log.i(TAG, "TFLite backend: GPU")
                    return InterpreterBundle(gpuInterpreter, delegate)
                }
                delegate.close()
            }
        }

        model.rewind()
        val cpuInterpreter = Interpreter(
            model,
            Interpreter.Options().apply { numThreads = 4 }
        )
        Log.i(TAG, "TFLite backend: CPU/XNNPACK")
        return InterpreterBundle(cpuInterpreter, null)
    }

    private fun loadModel(context: Context, assetName: String): ByteBuffer {
        val bytes = context.assets.open(assetName).use { it.readBytes() }
        require(bytes.isNotEmpty()) { "模型文件为空：$assetName" }
        return ByteBuffer
            .allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .apply {
                put(bytes)
                rewind()
            }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null
    }
}
