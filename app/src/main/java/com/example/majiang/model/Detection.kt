package com.example.majiang.model

import android.graphics.RectF

/** 一个检测结果：全图坐标下的框 + 类别 + 置信度 */
data class Detection(val box: RectF, val classId: Int, val score: Float)

object TileClasses {
    const val SUITED_CLASS_COUNT = 27
    const val FACE_CLASS_COUNT = SUITED_CLASS_COUNT
    const val CLASS_COUNT = FACE_CLASS_COUNT + 1
    const val BACK_ID = FACE_CLASS_COUNT
    const val EXPECTED_PER_TILE = 4
    const val DEFAULT_EXPECTED_TOTAL = 108

    /** 27 种牌面 + back，共 28 类 */
    val NAMES: Array<String> = Array(CLASS_COUNT) { id ->
        when {
            id < 9 -> "wan${id + 1}"
            id < 18 -> "tong${id - 8}"
            id < SUITED_CLASS_COUNT -> "tiao${id - 17}"
            else -> "back"
        }
    }

    private val DIGITS = arrayOf("一", "二", "三", "四", "五", "六", "七", "八", "九")

    /** 报告用中文牌名 */
    fun displayName(classId: Int): String = when {
        classId in 0 until 9 -> "${DIGITS[classId]}万"
        classId in 9 until 18 -> "${DIGITS[classId - 9]}筒"
        classId in 18 until SUITED_CLASS_COUNT -> "${DIGITS[classId - 18]}条"
        classId == BACK_ID -> "背面"
        else -> "未知"
    }

    /** 花色名：万/筒/条；back 返回 null */
    fun suitName(classId: Int): String? = when {
        classId in 0 until 9 -> "万"
        classId in 9 until 18 -> "筒"
        classId in 18 until SUITED_CLASS_COUNT -> "条"
        else -> null
    }

    fun isFaceId(classId: Int): Boolean = classId in 0 until FACE_CLASS_COUNT

    fun isValidId(classId: Int): Boolean = classId in NAMES.indices
}
