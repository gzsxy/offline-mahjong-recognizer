package com.example.majiang.ml

import android.graphics.Bitmap

/**
 * 把高分辨率原图切成相互重叠的固定大小块（默认 640×640，重叠 25%），
 * 用于密集小目标检测。每块保留在原图中的偏移量，便于坐标映射回去。
 */
class ImageSlicer(
    private val tileSize: Int = 640,
    private val overlap: Float = 0.25f
) {
    data class Slice(val offsetX: Int, val offsetY: Int, val bitmap: Bitmap)

    init {
        require(tileSize > 0) { "切片尺寸必须大于 0" }
        require(overlap >= 0f && overlap < 1f) { "切片重叠比例必须在 0 到 1 之间" }
    }

    fun slice(src: Bitmap): List<Slice> {
        require(!src.isRecycled) { "不能切分已回收的 Bitmap" }
        val xs = positionsFor(src.width)
        val ys = positionsFor(src.height)
        val out = ArrayList<Slice>(xs.size * ys.size)
        for (y in ys) {
            for (x in xs) {
                val w = minOf(tileSize, src.width - x)
                val h = minOf(tileSize, src.height - y)
                out.add(Slice(x, y, Bitmap.createBitmap(src, x, y, w, h)))
            }
        }
        return out
    }

    internal fun positionsFor(total: Int): List<Int> {
        require(total > 0) { "图像尺寸必须大于 0" }
        if (total <= tileSize) return listOf(0)
        val stride = (tileSize * (1f - overlap)).toInt().coerceAtLeast(1)
        val last = total - tileSize
        val list = mutableListOf<Int>()
        var position = 0
        while (true) {
            list += position.coerceAtMost(last)
            if (position >= last) break
            position += stride
        }
        return list.distinct().sorted()
    }
}
