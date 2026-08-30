package com.example.majiang.history

import com.example.majiang.history.HistoryStore.HistoryRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class HistoryStoreTest {

    @get:Rule
    val tmp = TemporaryFolder()

    private fun store(): HistoryStore = HistoryStore(tmp.newFolder("history"))

    private fun record(id: Long, total: Int = 108) = HistoryRecord(
        id = id,
        expected = 108,
        total = total,
        backCount = if (total < 108) 1 else 0,
        faceCount = total - if (total < 108) 1 else 0,
        suitCounts = mapOf("万" to 36, "筒" to 36, "条" to total - 72 - if (total < 108) 1 else 0),
        lowConfidenceCount = 0,
        lines = listOf("共检测到 $total 张", "万 36 · 筒 36 · 条 ${total - 72}")
    )

    @Test
    fun saveThenListRoundTripsAllFields() {
        val store = store()
        val original = record(id = 1_000_000L, total = 107)

        assertEquals(original, store.save(original, thumbnail = null))
        assertEquals(listOf(original), store.list())
    }

    @Test
    fun listIsSortedByNewestFirst() {
        val store = store()
        store.save(record(300L), thumbnail = null)
        store.save(record(100L), thumbnail = null)
        store.save(record(200L), thumbnail = null)

        assertEquals(listOf(300L, 200L, 100L), store.list().map { it.id })
    }

    @Test
    fun prunesOldestBeyondLimitWithThumbnails() {
        val store = store()
        (1..HistoryStore.MAX_RECORDS + 5).forEach { id ->
            store.save(record(id.toLong()), thumbnail = null)
            // 模拟缩略图文件存在
            val imageDir = tmp.root.resolve("history/img")
            imageDir.mkdirs()
            File(imageDir, "$id.jpg").writeBytes(byteArrayOf(1))
        }

        val listed = store.list()

        assertEquals(HistoryStore.MAX_RECORDS, listed.size)
        // 最旧的 1..5 号被清理，6 号成为最旧保留
        assertEquals(
            (6L..HistoryStore.MAX_RECORDS + 5L).toList(),
            listed.map { it.id }.reversed()
        )
        val imageDir = tmp.root.resolve("history/img")
        assertFalse(File(imageDir, "1.jpg").exists())
        assertFalse(File(imageDir, "5.jpg").exists())
        assertTrue(File(imageDir, "6.jpg").exists())
    }

    @Test
    fun deleteRemovesRecordAndThumbnail() {
        val store = store()
        store.save(record(42L), thumbnail = null)
        val imageDir = tmp.root.resolve("history/img")
        imageDir.mkdirs()
        val thumb = File(imageDir, "42.jpg")
        thumb.writeBytes(byteArrayOf(1, 2, 3))

        store.delete(42L)

        assertTrue(store.list().isEmpty())
        assertFalse(thumb.exists())
    }

    @Test
    fun clearRemovesAllRecordsAndThumbnails() {
        val store = store()
        store.save(record(1L), thumbnail = null)
        store.save(record(2L), thumbnail = null)
        val imageDir = tmp.root.resolve("history/img")
        imageDir.mkdirs()
        File(imageDir, "1.jpg").writeBytes(byteArrayOf(1))
        File(imageDir, "2.jpg").writeBytes(byteArrayOf(1))

        store.clear()

        assertTrue(store.list().isEmpty())
        assertTrue(imageDir.listFiles().isNullOrEmpty())
    }

    @Test
    fun emptyStoreListsNothing() {
        assertTrue(store().list().isEmpty())
    }
}
