package com.example.majiang.history

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.example.majiang.domain.TileAnalyzer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

/**
 * 识别历史本地存储（开发文档 FR-9 标准版）：
 * records.json 保存记录文本，img/<id>.jpg 保存结果缩略图，全部位于应用私有目录。
 * 任何存储失败都不抛出——历史记录是附属功能，不允许影响识别主流程。
 */
class HistoryStore(private val baseDir: File) {

    data class HistoryRecord(
        val id: Long,
        val expected: Int,
        val total: Int,
        val backCount: Int,
        val faceCount: Int,
        val suitCounts: Map<String, Int>,
        val lowConfidenceCount: Int,
        val lines: List<String>
    )

    private val recordsFile: File
        get() = File(baseDir, "records.json")
    private val imageDir: File
        get() = File(baseDir, "img")

    /** 保存一条记录；thumbnail 为标注结果图（可空）。成功返回记录本身，失败返回 null。 */
    fun save(record: HistoryRecord, thumbnail: Bitmap?): HistoryRecord? = try {
        baseDir.mkdirs()
        if (thumbnail != null) {
            writeThumbnail(record.id, thumbnail)
        }
        writeRecords(prune(readRecords() + record))
        record
    } catch (error: Throwable) {
        null
    }

    /** 全部记录，按时间倒序（最新在前）。 */
    fun list(): List<HistoryRecord> = try {
        readRecords().sortedByDescending { it.id }
    } catch (error: Throwable) {
        emptyList()
    }

    fun thumbnail(record: HistoryRecord): Bitmap? = try {
        val file = thumbnailFile(record.id)
        if (file.exists()) BitmapFactory.decodeFile(file.absolutePath) else null
    } catch (error: Throwable) {
        null
    }

    fun delete(id: Long) {
        try {
            writeRecords(readRecords().filterNot { it.id == id })
            thumbnailFile(id).delete()
        } catch (error: Throwable) {
            // 保留静默语义
        }
    }

    fun clear() {
        try {
            writeRecords(emptyList())
            imageDir.listFiles()?.forEach { it.delete() }
        } catch (error: Throwable) {
            // 保留静默语义
        }
    }

    /** 超出上限时丢弃最旧记录，并连带清理其缩略图。 */
    private fun prune(records: List<HistoryRecord>): List<HistoryRecord> {
        val sorted = records.sortedByDescending { it.id }
        sorted.drop(MAX_RECORDS).forEach { thumbnailFile(it.id).delete() }
        return sorted.take(MAX_RECORDS)
    }

    private fun thumbnailFile(id: Long): File = File(imageDir, "$id.jpg")

    private fun writeThumbnail(id: Long, bitmap: Bitmap) {
        imageDir.mkdirs()
        val scale = THUMBNAIL_MAX_SIZE / maxOf(bitmap.width, bitmap.height).toFloat()
        val scaled = if (scale < 1f) {
            Bitmap.createScaledBitmap(
                bitmap,
                (bitmap.width * scale).toInt().coerceAtLeast(1),
                (bitmap.height * scale).toInt().coerceAtLeast(1),
                true
            )
        } else {
            bitmap
        }
        FileOutputStream(thumbnailFile(id)).use { out ->
            scaled.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
        }
    }

    private fun readRecords(): List<HistoryRecord> {
        if (!recordsFile.exists()) return emptyList()
        val root = JSONObject(recordsFile.readText())
        val array = root.optJSONArray("records") ?: return emptyList()
        return (0 until array.length()).map { index ->
            val item = array.getJSONObject(index)
            HistoryRecord(
                id = item.getLong("id"),
                expected = item.getInt("expected"),
                total = item.getInt("total"),
                backCount = item.getInt("backCount"),
                faceCount = item.getInt("faceCount"),
                suitCounts = item.getJSONObject("suitCounts").let { suits ->
                    suits.keys().asSequence().associateWith { suits.getInt(it) }
                },
                lowConfidenceCount = item.getInt("lowConfidenceCount"),
                lines = item.getJSONArray("lines").let { lines ->
                    (0 until lines.length()).map { lines.getString(it) }
                }
            )
        }
    }

    private fun writeRecords(records: List<HistoryRecord>) {
        baseDir.mkdirs()
        val root = JSONObject()
        val array = JSONArray()
        records.forEach { record ->
            array.put(
                JSONObject()
                    .put("id", record.id)
                    .put("expected", record.expected)
                    .put("total", record.total)
                    .put("backCount", record.backCount)
                    .put("faceCount", record.faceCount)
                    .put(
                        "suitCounts",
                        JSONObject().apply {
                            record.suitCounts.forEach { (name, count) -> put(name, count) }
                        }
                    )
                    .put("lowConfidenceCount", record.lowConfidenceCount)
                    .put("lines", JSONArray(record.lines))
            )
        }
        root.put("records", array)
        // 先写临时文件再原子替换，避免写一半损坏记录
        val tmp = File(baseDir, "records.json.tmp")
        tmp.writeText(root.toString())
        tmp.renameTo(recordsFile)
    }

    companion object {
        const val MAX_RECORDS = 50
        const val THUMBNAIL_MAX_SIZE = 720
        const val JPEG_QUALITY = 80

        /** 从识别报告构造一条历史记录 */
        fun newRecord(id: Long, report: TileAnalyzer.Report): HistoryRecord = HistoryRecord(
            id = id,
            expected = report.expected,
            total = report.total,
            backCount = report.backCount,
            faceCount = report.faceCount,
            suitCounts = report.suitCounts,
            lowConfidenceCount = report.lowConfidenceCount,
            lines = report.lines
        )
    }
}
