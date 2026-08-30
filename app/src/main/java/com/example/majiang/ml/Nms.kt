package com.example.majiang.ml

import android.graphics.RectF
import com.example.majiang.model.Detection

/** 按类别分组的 IoU NMS。 */
object Nms {

    fun iou(a: RectF, b: RectF): Float {
        return iou(a.left, a.top, a.right, a.bottom, b.left, b.top, b.right, b.bottom)
    }

    /** 纯几何版本，便于在 JVM 单元测试中验证而不依赖 Android Canvas。 */
    internal fun iou(
        aLeft: Float,
        aTop: Float,
        aRight: Float,
        aBottom: Float,
        bLeft: Float,
        bTop: Float,
        bRight: Float,
        bBottom: Float
    ): Float {
        if (listOf(aLeft, aTop, aRight, aBottom, bLeft, bTop, bRight, bBottom).any { !it.isFinite() }) {
            return 0f
        }
        val x0 = maxOf(aLeft, bLeft)
        val y0 = maxOf(aTop, bTop)
        val x1 = minOf(aRight, bRight)
        val y1 = minOf(aBottom, bBottom)
        if (x1 <= x0 || y1 <= y0) return 0f
        val inter = (x1 - x0) * (y1 - y0)
        val aArea = (aRight - aLeft) * (aBottom - aTop)
        val bArea = (bRight - bLeft) * (bBottom - bTop)
        val union = aArea + bArea - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun apply(
        detections: List<Detection>,
        iouThreshold: Float,
        classAware: Boolean = true
    ): List<Detection> {
        require(iouThreshold in 0f..1f) { "IoU 阈值必须在 0 到 1 之间" }
        val kept = mutableListOf<Detection>()
        val groups = if (classAware) {
            detections.groupBy { it.classId }.values
        } else {
            listOf(detections)
        }
        groups.forEach { group ->
            val sorted = group.sortedByDescending { it.score }.toMutableList()
            while (sorted.isNotEmpty()) {
                val best = sorted.removeAt(0)
                kept.add(best)
                sorted.removeAll { iou(best.box, it.box) > iouThreshold }
            }
        }
        return kept.sortedByDescending { it.score }
    }

}
