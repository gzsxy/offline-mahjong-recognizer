package com.example.majiang.domain

import com.example.majiang.model.Detection
import com.example.majiang.model.TileClasses

/**
 * 业务规则引擎（开发文档 3.3）：
 * 计数（含背面牌）、花色分布、缺牌/整门缺失、背面牌智能提示。
 */
object TileAnalyzer {

    data class Report(
        val total: Int,
        val expected: Int,
        val backCount: Int,
        val faceCount: Int,
        val suitCounts: Map<String, Int>,          // 万/筒/条 各多少张（正面）
        val perTileCounts: List<Int>,              // 34 种正面牌各几张
        val shortage: List<Pair<Int, Int>>,        // (classId, 缺几张)
        val excess: List<Pair<Int, Int>>,          // (classId, 多几张)
        val missingSuits: List<String>,            // 整门缺失的花色
        val lowConfidenceCount: Int,               // 低置信检测数（建议重拍）
        val lines: List<String>                    // 给用户看的中文报告行
    )

    fun analyze(
        detections: List<Detection>,
        expected: Int = TileClasses.DEFAULT_EXPECTED_TOTAL,
        lowConfThreshold: Float = 0.55f
    ): Report {
        require(lowConfThreshold in 0f..1f) { "低置信度阈值必须在 0 到 1 之间" }
        val safeExpected = expected.coerceAtLeast(1)
        val back = detections.filter { it.classId == TileClasses.BACK_ID }
        val faces = detections.filter { TileClasses.isFaceId(it.classId) }

        val perTile = IntArray(TileClasses.FACE_CLASS_COUNT)
        faces.forEach { if (TileClasses.isFaceId(it.classId)) perTile[it.classId]++ }

        val suitCounts = linkedMapOf("万" to 0, "筒" to 0, "条" to 0)
        faces.forEach { d ->
            TileClasses.suitName(d.classId)?.let { s -> suitCounts[s] = suitCounts[s]!! + 1 }
        }

        val shortage = (0 until TileClasses.FACE_CLASS_COUNT)
            .map { it to TileClasses.EXPECTED_PER_TILE - perTile[it] }
            .filter { it.second > 0 }
        val excess = (0 until TileClasses.FACE_CLASS_COUNT)
            .map { it to perTile[it] - TileClasses.EXPECTED_PER_TILE }
            .filter { it.second > 0 }
        val missingSuits = listOf(
            "万" to (0 until 9),
            "筒" to (9 until 18),
            "条" to (18 until TileClasses.SUITED_CLASS_COUNT)
        ).filter { (_, range) -> range.all { perTile[it] == 0 } }.map { it.first }

        val total = detections.size
        val lowConf = detections.count { it.score < lowConfThreshold }

        // ---- 报告文本 ----
        val lines = mutableListOf<String>()
        lines += when {
            total == safeExpected -> "共检测到 $total 张 ✓ 数量正确"
            total < safeExpected -> "数量不足：检测到 $total 张，缺 ${safeExpected - total} 张"
            else -> "数量超出：检测到 $total 张，多 ${total - safeExpected} 张"
        }
        if (safeExpected < TileClasses.DEFAULT_EXPECTED_TOTAL) {
            lines += "当前为部分清点（期望 $safeExpected 张），缺牌明细按整副 ${TileClasses.DEFAULT_EXPECTED_TOTAL} 张口径计算"
        }
        lines += buildString {
            append("万 ${suitCounts["万"]} · 筒 ${suitCounts["筒"]} · 条 ${suitCounts["条"]}")
            if (back.isNotEmpty()) append(" · 背面 ${back.size}")
        }
        if (missingSuits.isNotEmpty()) {
            lines += "整门缺失：${missingSuits.joinToString("、")}"
        }
        if (shortage.isNotEmpty()) {
            lines += "缺牌：" + shortage.joinToString("、") {
                "${TileClasses.displayName(it.first)} ×${it.second}"
            }
        } else if (missingSuits.isEmpty()) {
            lines += "${TileClasses.FACE_CLASS_COUNT} 种牌齐全"
        }
        if (excess.isNotEmpty()) {
            lines += "多出：" + excess.joinToString("、") {
                "${TileClasses.displayName(it.first)} ×${it.second}"
            }
        }
        // 智能提示（文档 3.3-3）
        if (back.isNotEmpty() && total == safeExpected && (shortage.isNotEmpty() || missingSuits.isNotEmpty())) {
            val what = buildList {
                addAll(missingSuits.map { "整门$it" })
                addAll(shortage.map { "${TileClasses.displayName(it.first)}×${it.second}" })
            }.joinToString("、")
            lines += "提示：检测到 ${back.size} 张牌花色朝下，缺少的【$what】可能就在其中，请翻开核对"
        } else if (back.isNotEmpty() && total != safeExpected) {
            lines += "提示：检测到 ${back.size} 张牌花色朝下，请翻开后再确认数量与花色"
        }
        if (lowConf > 0) {
            lines += "注意：$lowConf 张牌识别置信度较低，建议换个角度重拍确认"
        }

        return Report(
            total = total,
            expected = safeExpected,
            backCount = back.size,
            faceCount = faces.size,
            suitCounts = suitCounts,
            perTileCounts = perTile.toList(),
            shortage = shortage,
            excess = excess,
            missingSuits = missingSuits,
            lowConfidenceCount = lowConf,
            lines = lines
        )
    }
}
