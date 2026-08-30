package com.example.majiang.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ImageSlicerTest {

    @Test
    fun smallImageUsesOneSlice() {
        assertEquals(listOf(0), ImageSlicer(640, 0.25f).positionsFor(640))
        assertEquals(listOf(0), ImageSlicer(640, 0.25f).positionsFor(500))
    }

    @Test
    fun largeImageEndsAtTheLastValidOffsetWithoutExtraSlice() {
        val positions = ImageSlicer(640, 0.25f).positionsFor(1000)

        assertEquals(listOf(0, 360), positions)
        assertTrue(positions.all { it in 0..360 })
    }

    @Test(expected = IllegalArgumentException::class)
    fun invalidOverlapIsRejected() {
        ImageSlicer(640, 1f)
    }

    @Test
    fun cameraResolutionAt1280ProducesTwelveSlices() {
        val slicer = ImageSlicer(1280, 0.25f)
        // 4000x3000 相机原图：x 轴 4 个位置，y 轴 3 个位置，共 12 块
        assertEquals(listOf(0, 960, 1920, 2720), slicer.positionsFor(4000))
        assertEquals(listOf(0, 960, 1720), slicer.positionsFor(3000))
        assertEquals(12, slicer.positionsFor(4000).size * slicer.positionsFor(3000).size)
    }
}
