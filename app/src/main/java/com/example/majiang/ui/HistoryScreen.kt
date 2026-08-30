package com.example.majiang.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.majiang.history.HistoryStore
import com.example.majiang.history.HistoryStore.HistoryRecord
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HistoryListScreen(
    store: HistoryStore,
    onOpen: (HistoryRecord) -> Unit,
    onBack: () -> Unit
) {
    val records by produceState<List<HistoryRecord>>(emptyList()) {
        value = withContext(Dispatchers.IO) { store.list() }
    }
    var confirmClear by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize()) {
        HistoryHeader(
            title = "历史记录",
            onBack = onBack,
            action = if (records.isEmpty()) null else "清空全部",
            onAction = { confirmClear = true }
        )
        if (records.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("暂无历史记录", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            LazyColumn(Modifier.fillMaxSize()) {
                items(records, key = { it.id }) { record ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onOpen(record) }
                            .padding(horizontal = 16.dp, vertical = 10.dp)
                    ) {
                        HistoryThumbnail(
                            store = store,
                            record = record,
                            modifier = Modifier
                                .width(88.dp)
                                .height(66.dp)
                        )
                        Spacer(Modifier.width(12.dp))
                        Column {
                            Text(
                                formatTime(record.id),
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Spacer(Modifier.height(4.dp))
                            Text(
                                "总数 ${record.total} / 期望 ${record.expected}",
                                style = MaterialTheme.typography.bodyLarge
                            )
                            record.lines.firstOrNull {
                                it.contains("缺牌") || it.contains("整门缺失")
                            }?.let {
                                Spacer(Modifier.height(4.dp))
                                Text(
                                    it,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.error,
                                    maxLines = 1
                                )
                            }
                        }
                    }
                    HorizontalDivider()
                }
            }
        }
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("清空历史记录") },
            text = { Text("将删除全部 ${records.size} 条记录及缩略图，且无法恢复。确定继续吗？") },
            confirmButton = {
                TextButton(onClick = {
                    confirmClear = false
                    store.clear()
                    onBack()
                }) { Text("清空") }
            },
            dismissButton = {
                TextButton(onClick = { confirmClear = false }) { Text("取消") }
            }
        )
    }
}

@Composable
fun HistoryDetailScreen(
    store: HistoryStore,
    record: HistoryRecord,
    onDelete: () -> Unit,
    onBack: () -> Unit
) {
    val thumbnail by produceState<Bitmap?>(null, record.id) {
        value = withContext(Dispatchers.IO) { store.thumbnail(record) }
    }

    Column(Modifier.fillMaxSize()) {
        HistoryHeader(title = "记录详情", onBack = onBack, action = "删除", onAction = onDelete)
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp)
        ) {
            Text(
                formatTime(record.id),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(8.dp))
            val image = thumbnail
            if (image != null) {
                Image(
                    bitmap = image.asImageBitmap(),
                    contentDescription = "识别结果缩略图",
                    contentScale = ContentScale.FillWidth,
                    modifier = Modifier.fillMaxWidth()
                )
            } else {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .height(120.dp)
                        .background(MaterialTheme.colorScheme.surfaceVariant),
                    contentAlignment = Alignment.Center
                ) {
                    Text("无缩略图", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Spacer(Modifier.height(12.dp))
            record.lines.forEach {
                Text(it, style = MaterialTheme.typography.bodyLarge)
                Spacer(Modifier.height(6.dp))
            }
        }
    }
}

@Composable
private fun HistoryHeader(
    title: String,
    onBack: () -> Unit,
    action: String?,
    onAction: () -> Unit
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 6.dp)
    ) {
        TextButton(onClick = onBack) { Text("返回") }
        Text(
            title,
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.weight(1f)
        )
        if (action != null) {
            OutlinedButton(onClick = onAction) { Text(action, fontSize = 14.sp) }
        }
        Spacer(Modifier.width(8.dp))
    }
}

@Composable
private fun HistoryThumbnail(
    store: HistoryStore,
    record: HistoryRecord,
    modifier: Modifier
) {
    val bitmap by produceState<Bitmap?>(null, record.id) {
        value = withContext(Dispatchers.IO) { store.thumbnail(record) }
    }
    val image = bitmap
    if (image != null) {
        Image(
            bitmap = image.asImageBitmap(),
            contentDescription = "识别结果缩略图",
            contentScale = ContentScale.Crop,
            modifier = modifier
        )
    } else {
        Box(
            modifier.background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            Text("无图", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 12.sp)
        }
    }
}

private fun formatTime(timestamp: Long): String =
    SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(timestamp))
