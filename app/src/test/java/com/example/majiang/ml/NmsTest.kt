package com.example.majiang.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class NmsTest {

    @Test
    fun iouForIdenticalBoxesIsOne() {
        assertEquals(
            1f,
            Nms.iou(0f, 0f, 10f, 10f, 0f, 0f, 10f, 10f),
            0.0001f
        )
    }

    @Test
    fun iouForDisjointOrInvalidBoxesIsZero() {
        assertEquals(
            0f,
            Nms.iou(0f, 0f, 10f, 10f, 20f, 20f, 30f, 30f),
            0.0001f
        )
        assertEquals(
            0f,
            Nms.iou(Float.NaN, 0f, 10f, 10f, 0f, 0f, 10f, 10f),
            0.0001f
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun invalidThresholdIsRejected() {
        Nms.apply(emptyList(), 1.1f)
    }

    @Test
    fun containmentGeometryDistinguishesNestedFromAdjacent() {
        // JVM 单测中 android.jar 的 RectF 字段不生效，因此直接验证纯几何版本
        val nested = Nms.containment(0f, 0f, 100f, 200f, 10f, 40f, 90f, 160f)
        assertEquals(1f, nested, 0.001f)

        val partial = Nms.containment(0f, 0f, 100f, 200f, 50f, 0f, 150f, 100f)
        assertEquals(0.5f, partial, 0.001f)

        val adjacent = Nms.containment(0f, 0f, 100f, 200f, 110f, 0f, 210f, 200f)
        assertEquals(0f, adjacent, 0.001f)
    }
}
