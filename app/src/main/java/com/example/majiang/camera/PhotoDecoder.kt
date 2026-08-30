package com.example.majiang.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import kotlin.math.roundToInt

/** 解码本地文件或相册 Uri，并把 EXIF 方向真正应用到像素。 */
object PhotoDecoder {

    /** 准确率优先：只对超过上限的极端大图采样，避免小牌漏检。 */
    fun decode(file: File, maxDimension: Int = 4096): Bitmap? = decode(
        openStream = { FileInputStream(file) },
        readOrientation = {
            runCatching {
                ExifInterface(file).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
        },
        maxDimension = maxDimension
    )

    /** 使用系统相册返回的 Uri 解码，不需要申请存储权限。 */
    fun decode(context: Context, uri: Uri, maxDimension: Int = 4096): Bitmap? = decode(
        openStream = { context.contentResolver.openInputStream(uri) },
        readOrientation = {
            runCatching {
                context.contentResolver.openInputStream(uri)?.use { input ->
                    ExifInterface(input).getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                } ?: ExifInterface.ORIENTATION_NORMAL
            }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
        },
        maxDimension = maxDimension
    )

    private fun decode(
        openStream: () -> InputStream?,
        readOrientation: () -> Int,
        maxDimension: Int
    ): Bitmap? {
        require(maxDimension > 0) { "处理图最长边必须大于 0" }

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        openStream()?.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        var sampleSize = 1
        while (maxOf(bounds.outWidth / sampleSize, bounds.outHeight / sampleSize) > maxDimension) {
            sampleSize *= 2
        }
        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val source = openStream()?.use { BitmapFactory.decodeStream(it, null, options) } ?: return null
        val matrix = Matrix()
        when (readOrientation()) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.setScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.setRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.setRotate(-90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(-90f)
            else -> Unit
        }

        val oriented = runCatching {
            Bitmap.createBitmap(
                source,
                0,
                0,
                source.width,
                source.height,
                matrix,
                true
            )
        }.getOrElse { source }
        if (oriented !== source && !source.isRecycled) source.recycle()

        if (maxOf(oriented.width, oriented.height) <= maxDimension) return oriented
        val scale = maxDimension.toFloat() / maxOf(oriented.width, oriented.height)
        val resized = Bitmap.createScaledBitmap(
            oriented,
            (oriented.width * scale).roundToInt().coerceAtLeast(1),
            (oriented.height * scale).roundToInt().coerceAtLeast(1),
            true
        )
        if (resized !== oriented && !oriented.isRecycled) oriented.recycle()
        return resized
    }
}
