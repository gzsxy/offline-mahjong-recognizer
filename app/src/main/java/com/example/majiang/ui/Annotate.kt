package com.example.majiang.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import com.example.majiang.model.Detection
import com.example.majiang.model.TileClasses

/** 在原图画布副本上绘制检测框与牌名（正面=绿，背面=黄）。 */
fun annotateDetections(src: Bitmap, detections: List<Detection>): Bitmap {
    val out = src.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val strokeW = (src.width / 300f).coerceAtLeast(2f)
    val textSize = (src.width / 70f).coerceAtLeast(24f)

    val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = strokeW
    }
    val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        alpha = 160
    }
    val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.textSize = textSize
        color = Color.BLACK
        isFakeBoldText = true
    }

    for (d in detections) {
        val isBack = d.classId == TileClasses.BACK_ID
        val color = if (isBack) Color.YELLOW else Color.GREEN
        boxPaint.color = color
        canvas.drawRect(d.box, boxPaint)

        val label = TileClasses.displayName(d.classId)
        val tw = textPaint.measureText(label)
        val ty = if (d.box.top - textSize - 6 >= 0) d.box.top - 6 else d.box.bottom + textSize + 4
        textBgPaint.color = color
        canvas.drawRect(d.box.left, ty - textSize - 4, d.box.left + tw + 8, ty + 4, textBgPaint)
        canvas.drawText(label, d.box.left + 4, ty - 2, textPaint)
    }
    return out
}
