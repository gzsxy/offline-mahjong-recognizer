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
}
