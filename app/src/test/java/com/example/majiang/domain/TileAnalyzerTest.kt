package com.example.majiang.domain

import android.graphics.RectF
import com.example.majiang.model.Detection
import com.example.majiang.model.TileClasses
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TileAnalyzerTest {

    private val box = RectF(0f, 0f, 10f, 10f)

    @Test
    fun completeSetIsReportedAsComplete() {
        val detections = buildList {
            repeat(TileClasses.FACE_CLASS_COUNT) { classId ->
                repeat(TileClasses.EXPECTED_PER_TILE) {
                    add(Detection(box, classId, 0.9f))
                }
            }
        }

        val report = TileAnalyzer.analyze(detections, expected = TileClasses.FACE_CLASS_COUNT * 4)

        assertEquals(TileClasses.FACE_CLASS_COUNT * 4, report.total)
        assertEquals(TileClasses.FACE_CLASS_COUNT * 4, report.faceCount)
        assertTrue(report.shortage.isEmpty())
        assertTrue(report.excess.isEmpty())
        assertTrue(report.lines.any { it.contains("数量正确") })
        assertTrue(report.lines.any { it.contains("27 种牌齐全") })
    }

    @Test
    fun partialCountAddsExplanationLine() {
        val detections = buildList {
            repeat(9) { classId -> repeat(4) { add(Detection(box, classId, 0.9f)) } }
        }

        val report = TileAnalyzer.analyze(detections, expected = 36)

        assertEquals(36, report.total)
        assertTrue(report.lines.any { it.contains("部分清点") && it.contains("36 张") })
        assertTrue(report.shortage.isNotEmpty())
    }

    @Test
    fun backTilesExplainPossibleShortage() {
        val detections = buildList {
            repeat(3) { add(Detection(box, 0, 0.9f)) }
            repeat(TileClasses.FACE_CLASS_COUNT - 1) { classId ->
                repeat(TileClasses.EXPECTED_PER_TILE) {
                    add(Detection(box, classId + 1, 0.9f))
                }
            }
            add(Detection(box, TileClasses.BACK_ID, 0.9f))
        }

        val report = TileAnalyzer.analyze(detections)

        assertEquals(TileClasses.FACE_CLASS_COUNT * 4, report.total)
        assertEquals(1, report.backCount)
        assertEquals(3, report.perTileCounts[0])
        assertTrue(report.lines.any { it.contains("朝下") && it.contains("翻开") })
    }

    @Test
    fun invalidClassDoesNotCrashDisplayHelpers() {
        assertEquals("未知", TileClasses.displayName(-1))
        assertEquals("未知", TileClasses.displayName(TileClasses.CLASS_COUNT))
        assertEquals(null, TileClasses.suitName(-1))
        assertEquals(null, TileClasses.suitName(TileClasses.BACK_ID))
    }
}
