package com.example.majiang.ml

import android.graphics.Bitmap
import android.graphics.RectF
import com.example.majiang.model.Detection
import com.example.majiang.model.TileClasses
import java.io.Closeable

/**
 * 切片 → 逐块检测 → 坐标映射回原图 → 全局 NMS 合并。
 */
class AnalysisPipeline(
    private val detector: DetectionBackend,
    private val slicer: ImageSlicer = ImageSlicer(),
    private val sliceNmsIou: Float = 0.45f,
    private val globalNmsIou: Float = 0.5f,
    // 切片接缝处的重复框 IoU 可能低于 0.35，相邻真实牌框几乎不重叠（IoU≈0），
    // 收紧到 0.3 可在不动真实框的前提下去掉接缝重复。
    private val faceGlobalNmsIou: Float = 0.3f,
    private val backGlobalNmsIou: Float = 0.3f
) : Closeable {

    /** 进度回调：已完成切片数 / 总切片数 */
    var onProgress: ((done: Int, total: Int) -> Unit)? = null

    init {
        require(faceGlobalNmsIou in 0f..1f) { "正面 IoU 阈值必须在 0 到 1 之间" }
        require(backGlobalNmsIou in 0f..1f) { "牌背 IoU 阈值必须在 0 到 1 之间" }
    }

    @Synchronized
    fun analyze(fullImage: Bitmap): List<Detection> {
        require(!fullImage.isRecycled) { "不能分析已回收的 Bitmap" }
        val slices = slicer.slice(fullImage)
        val all = ArrayList<Detection>(512)
        try {
            slices.forEachIndexed { index, slice ->
                try {
                    val dets = detector.detect(slice.bitmap)
                    // 块内 NMS 先压缩候选数，再做全局合并
                    Nms.apply(dets, sliceNmsIou).forEach { d ->
                        val b = d.box
                        all.add(
                            Detection(
                                RectF(
                                    b.left + slice.offsetX, b.top + slice.offsetY,
                                    b.right + slice.offsetX, b.bottom + slice.offsetY
                                ),
                                d.classId, d.score
                            )
                        )
                    }
                    onProgress?.invoke(index + 1, slices.size)
                } finally {
                    if (!slice.bitmap.isRecycled) slice.bitmap.recycle()
                }
            }
            val faces = all.filter { it.classId != TileClasses.BACK_ID }
            val backs = all.filter { it.classId == TileClasses.BACK_ID }
            return (Nms.apply(faces, minOf(globalNmsIou, faceGlobalNmsIou)) +
                Nms.apply(backs, minOf(globalNmsIou, backGlobalNmsIou)))
                .sortedByDescending { it.score }
        } finally {
            // 出现模型异常时也要释放尚未处理的切片。
            slices.forEach { if (!it.bitmap.isRecycled) it.bitmap.recycle() }
        }
    }

    @Synchronized
    override fun close() {
        detector.close()
    }
}
