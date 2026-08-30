package com.example.majiang.ml

import android.graphics.Bitmap
import android.graphics.RectF
import com.example.majiang.model.Detection
import com.example.majiang.model.TileClasses
import java.io.Closeable
import kotlin.math.ceil
import kotlin.math.floor

/** 单个或多个 TFLite 检测模型的统一接口。 */
interface DetectionBackend : Closeable {
    fun detect(slice: Bitmap): List<Detection>
}

/**
 * 正面与牌背使用不同训练域的模型，避免用牌背微调时遗忘正面牌能力。
 * 两个分支只保留各自负责的类别，再交给同一套切片和 NMS 处理。
 */
class FaceAndBackEnsemble(
    private val faceDetector: DetectionBackend,
    private val backDetector: DetectionBackend,
    private val scatteredBackDetector: DetectionBackend? = null
) : DetectionBackend {

    override fun detect(slice: Bitmap): List<Detection> = buildList {
        addAll(faceDetector.detect(slice).filter { it.classId != TileClasses.BACK_ID })

        val regularBacks = backDetector.detect(slice)
            .filter { it.classId == TileClasses.BACK_ID }
        addAll(regularBacks)

        // The scattered model is deliberately used as a fallback. The regular
        // model is stronger on dense grids, while the auxiliary model is more
        // sensitive to isolated colored backs and otherwise fires on white faces.
        if (regularBacks.isEmpty()) {
            addAll(
                scatteredBackDetector?.detect(slice)
                    ?.filter { it.classId == TileClasses.BACK_ID }
                    ?.filter { hasColoredBackAppearance(slice, it.box) }
                    .orEmpty()
            )
        }
    }

    override fun close() {
        try {
            faceDetector.close()
        } finally {
            try {
                backDetector.close()
            } finally {
                scatteredBackDetector?.close()
            }
        }
    }

    private fun hasColoredBackAppearance(bitmap: Bitmap, box: RectF): Boolean {
        if (bitmap.width == 0 || bitmap.height == 0) return false
        val left = floor(box.left).toInt().coerceIn(0, bitmap.width - 1)
        val top = floor(box.top).toInt().coerceIn(0, bitmap.height - 1)
        val right = ceil(box.right).toInt().coerceIn(left + 1, bitmap.width)
        val bottom = ceil(box.bottom).toInt().coerceIn(top + 1, bitmap.height)
        val width = right - left
        val height = bottom - top
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, left, top, width, height)

        var colored = 0
        for (pixel in pixels) {
            val red = (pixel shr 16) and 0xFF
            val green = (pixel shr 8) and 0xFF
            val blue = pixel and 0xFF
            val maximum = maxOf(red, green, blue)
            val minimum = minOf(red, green, blue)
            val spread = maximum - minimum
            if (maximum <= 0 || spread <= 25 || spread * 5 <= maximum) continue

            val blueBack = blue * 100 > red * 112 && blue * 100 > green * 103
            val greenBack = green * 100 > red * 112 && green * 100 > blue * 103
            if (blueBack || greenBack) colored++
        }
        return colored.toFloat() / pixels.size >= 0.18f
    }
}
