package com.example.majiang

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Surface
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.majiang.camera.PhotoDecoder
import com.example.majiang.domain.TileAnalyzer
import com.example.majiang.history.HistoryStore
import com.example.majiang.ml.AnalysisPipeline
import com.example.majiang.ml.FaceAndBackEnsemble
import com.example.majiang.ml.ImageSlicer
import com.example.majiang.ml.TileDetector
import com.example.majiang.model.TileClasses
import com.example.majiang.ui.HistoryDetailScreen
import com.example.majiang.ui.HistoryListScreen
import com.example.majiang.ui.annotateDetections
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val TAG = "MahjongApp"

// 调试回归触发器：进程级一次性标记，防止 Activity 重建或重组导致重复触发识别
private var debugTriggerConsumed = false

// 识别并发门闩：同一时刻只允许一个分析在跑（重复触发直接抑制）
private val analysisGate = java.util.concurrent.atomic.AtomicBoolean(false)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // 调试入口：adb shell am start ... --es debug_image /sdcard/xxx.jpg
        // 免交互直接进识别管线，用于真机自动化回归。
        setContent { MaterialTheme { AppRoot(debugImagePath = intent?.getStringExtra("debug_image")) } }
    }
}

sealed interface ScreenState {
    data object Camera : ScreenState
    data class Processing(val done: Int, val total: Int) : ScreenState
    data class Result(val image: Bitmap, val lines: List<String>) : ScreenState
    data object HistoryList : ScreenState
    data class HistoryDetail(val record: HistoryStore.HistoryRecord) : ScreenState
    data class Error(val message: String) : ScreenState
}

private sealed interface PhotoInput {
    data class Camera(val file: File) : PhotoInput
    data class Gallery(val uri: Uri) : PhotoInput
    /** 调试回归用：直接引用外部路径，识别后不删除。 */
    data class DebugFile(val file: File) : PhotoInput
}

private data class PendingPhoto(val input: PhotoInput, val expected: Int)

private sealed interface ModelState {
    data object Loading : ModelState
    data class Ready(val pipeline: AnalysisPipeline) : ModelState
    data class Failed(val message: String) : ModelState
}

@Composable
fun AppRoot(debugImagePath: String? = null) {
    val context = LocalContext.current
    val appContext = context.applicationContext
    val historyStore = remember { HistoryStore(File(appContext.filesDir, "history")) }
    var modelAttempt by remember { mutableStateOf(0) }
    var modelState by remember { mutableStateOf<ModelState>(ModelState.Loading) }
    var screen by remember { mutableStateOf<ScreenState>(ScreenState.Camera) }
    var expectedText by remember { mutableStateOf("${TileClasses.DEFAULT_EXPECTED_TOTAL}") }
    var pendingPhoto by remember { mutableStateOf<PendingPhoto?>(null) }

    LaunchedEffect(modelAttempt) {
        modelState = ModelState.Loading
        val loaded = withContext(Dispatchers.IO) {
            try {
                Result.success(createAnalysisPipeline(appContext))
            } catch (error: CancellationException) {
                throw error
            } catch (error: Exception) {
                Result.failure(error)
            }
        }
        modelState = loaded.fold(
            onSuccess = { ModelState.Ready(it) },
            onFailure = {
                ModelState.Failed(it.message ?: "模型加载失败")
            }
        )
    }

    val pipeline = (modelState as? ModelState.Ready)?.pipeline
    DisposableEffect(pipeline) {
        onDispose { pipeline?.close() }
    }

    when (val state = screen) {
        ScreenState.Camera -> {
            val modelError = (modelState as? ModelState.Failed)?.message
            CameraScreen(
                expectedText = expectedText,
                onExpectedChange = { expectedText = it.filter(Char::isDigit).take(3) },
                onPhotoTaken = { file ->
                    val expected = expectedText.toIntOrNull() ?: TileClasses.DEFAULT_EXPECTED_TOTAL
                    pendingPhoto = PendingPhoto(PhotoInput.Camera(file), expected)
                    screen = ScreenState.Processing(0, 0)
                },
                onImageSelected = { uri ->
                    val expected = expectedText.toIntOrNull() ?: TileClasses.DEFAULT_EXPECTED_TOTAL
                    pendingPhoto = PendingPhoto(PhotoInput.Gallery(uri), expected)
                    screen = ScreenState.Processing(0, 0)
                },
                modelReady = pipeline != null,
                modelError = modelError,
                onRetryModel = { modelAttempt++ },
                onOpenHistory = { screen = ScreenState.HistoryList }
            )
        }
        is ScreenState.Processing -> ProcessingScreen(state)
        is ScreenState.Result -> ResultScreen(
            image = state.image,
            lines = state.lines,
            onRetake = { screen = ScreenState.Camera }
        )
        ScreenState.HistoryList -> HistoryListScreen(
            store = historyStore,
            onOpen = { screen = ScreenState.HistoryDetail(it) },
            onBack = { screen = ScreenState.Camera }
        )
        is ScreenState.HistoryDetail -> HistoryDetailScreen(
            store = historyStore,
            record = state.record,
            onDelete = {
                historyStore.delete(state.record.id)
                screen = ScreenState.HistoryList
            },
            onBack = { screen = ScreenState.HistoryList }
        )
        is ScreenState.Error -> ErrorScreen(
            message = state.message,
            onRetake = { screen = ScreenState.Camera },
            onRetryModel = { modelAttempt++ }
        )
    }

    // 调试回归：冷启动带 debug_image 时自动进识别管线
    LaunchedEffect(debugImagePath, pipeline) {
        if (debugImagePath != null && !debugTriggerConsumed && pipeline != null &&
            pendingPhoto == null && screen == ScreenState.Camera
        ) {
            debugTriggerConsumed = true
            Log.w(TAG, "DEBUG TRIGGER fired, consumed=$debugTriggerConsumed")
            pendingPhoto = PendingPhoto(
                PhotoInput.DebugFile(File(debugImagePath)),
                expected = TileClasses.DEFAULT_EXPECTED_TOTAL
            )
            screen = ScreenState.Processing(0, 0)
        }
    }

    val mainHandler = remember { Handler(Looper.getMainLooper()) }
    // 识别消费循环：LaunchedEffect(Unit) 永不因 key 变化重启，避免长任务中途被
    // 取消丢结果；并发由 analysisGate 保证，轮询消费 pendingPhoto。
    val latestPendingPhoto by rememberUpdatedState(pendingPhoto)
    val latestPipeline by rememberUpdatedState(pipeline)
    LaunchedEffect(Unit) {
        while (true) {
            val request = latestPendingPhoto
            val activePipeline = latestPipeline
            if (request == null || activePipeline == null ||
                !analysisGate.compareAndSet(false, true)
            ) {
                delay(150)
                continue
            }

            try {
            val sourceDescription = when (val input = request.input) {
                is PhotoInput.Camera -> input.file.absolutePath
                is PhotoInput.Gallery -> input.uri.toString()
                is PhotoInput.DebugFile -> input.file.absolutePath
            }
            Log.i(TAG, "analysis started(#${System.identityHashCode(request)}): $sourceDescription")
            Log.i(TAG, "effect enter: pendingPhoto=${pendingPhoto?.hashCode()} pipeline=${activePipeline.hashCode()}")
            val result = withContext(Dispatchers.Default) {
                val bitmap = when (val input = request.input) {
                    is PhotoInput.Camera -> PhotoDecoder.decode(input.file)
                    is PhotoInput.Gallery -> PhotoDecoder.decode(appContext, input.uri)
                    is PhotoInput.DebugFile -> PhotoDecoder.decode(input.file)
                }
                    ?: error("无法读取拍摄的照片")
                try {
                    Log.i(TAG, "decoded bitmap: ${bitmap.width}x${bitmap.height}")
                    activePipeline.onProgress = { done, total ->
                        mainHandler.post {
                            if (pendingPhoto == request && screen is ScreenState.Processing) {
                                screen = ScreenState.Processing(done, total)
                            }
                        }
                    }
                    val detections = activePipeline.analyze(bitmap)
                    val report = TileAnalyzer.analyze(detections, request.expected)
                    val annotated = annotateDetections(bitmap, detections)
                    // 历史保存失败（返回 null）不影响识别结果展示
                    val saved = withContext(Dispatchers.IO) {
                        historyStore.save(
                            HistoryStore.newRecord(System.currentTimeMillis(), report),
                            annotated
                        )
                    }
                    Log.i(
                        TAG,
                        "analysis complete: detections=${detections.size}, historySaved=${saved != null}"
                    )
                    annotated to report.lines
                } finally {
                    if (!bitmap.isRecycled) bitmap.recycle()
                }
            }
            screen = ScreenState.Result(result.first, result.second)
        } catch (error: CancellationException) {
            Log.w(TAG, "analysis cancelled (leaving composition)")
            throw error
        } catch (error: Exception) {
            Log.e(TAG, "analysis failed", error)
            screen = ScreenState.Error("识别失败：${error.message ?: "未知错误"}")
        } finally {
            analysisGate.set(false)
            activePipeline.onProgress = null
            when (request.input) {
                is PhotoInput.Camera -> request.input.file.delete()
                is PhotoInput.Gallery, is PhotoInput.DebugFile -> Unit
            }
            pendingPhoto = null
        }
        delay(150)
        }
    }
}

private fun createAnalysisPipeline(context: android.content.Context): AnalysisPipeline {
    // B 阶段新一代模型：正面 YOLO11l@1280 + 牌背 YOLO11m@1280（开发文档 4.1-B）。
    // 1280 切片下切片数约为 640 方案的 1/4，散放牌背辅助模型由混合域训练的新牌背模型替代。
        // 混合精度：正面 int8（实测框质量无损、速度快）+ 牌背 FP32（int8 实测框回归劣化，见开发文档 10.6）。
    val faceDetector = TileDetector(
        context,
        assetName = "mahjong_11l_1280_int8.tflite",
        preferGpu = false,
        modelClassCount = 28
    )
    return try {
        val backDetector = TileDetector(
            context,
            assetName = "mahjong_11m_back_1280_fp32.tflite",
            preferGpu = false,
            modelClassCount = 28
        )
        try {
            AnalysisPipeline(
                FaceAndBackEnsemble(faceDetector, backDetector),
                slicer = ImageSlicer(tileSize = 1280)
            )
        } catch (error: Throwable) {
            backDetector.close()
            throw error
        }
    } catch (error: Throwable) {
        faceDetector.close()
        throw error
    }
}

@Composable
fun CameraScreen(
    expectedText: String,
    onExpectedChange: (String) -> Unit,
    onPhotoTaken: (File) -> Unit,
    onImageSelected: (Uri) -> Unit,
    modelReady: Boolean,
    modelError: String?,
    onRetryModel: () -> Unit,
    onOpenHistory: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    var cameraError by remember { mutableStateOf<String?>(null) }
    var previewView by remember { mutableStateOf<PreviewView?>(null) }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }
    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> uri?.let(onImageSelected) }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                hasPermission = ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    LaunchedEffect(hasPermission) {
        if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setJpegQuality(95)
            .build()
    }
    var busy by remember { mutableStateOf(false) }
    val expectedCount = expectedText.toIntOrNull()?.takeIf { it in 1..999 }

    Column(Modifier.fillMaxSize()) {
        if (hasPermission) {
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).also { view ->
                        previewView = view
                        val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                        cameraProviderFuture.addListener({
                            runCatching {
                                val cameraProvider = cameraProviderFuture.get()
                                val preview = Preview.Builder().build().also {
                                    it.setSurfaceProvider(view.surfaceProvider)
                                }
                                cameraProvider.unbindAll()
                                cameraProvider.bindToLifecycle(
                                    lifecycleOwner,
                                    CameraSelector.DEFAULT_BACK_CAMERA,
                                    preview,
                                    imageCapture
                                )
                                cameraError = null
                            }.onFailure {
                                cameraError = it.message ?: "相机初始化失败"
                            }
                        }, ContextCompat.getMainExecutor(ctx))
                    }
                },
                update = { previewView = it },
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            )
        } else {
            Box(
                Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                Text("需要相机权限才能拍摄")
            }
        }

        Column(Modifier.padding(16.dp)) {
            cameraError?.let {
                Text("相机不可用：$it", color = MaterialTheme.colorScheme.error)
                Spacer(Modifier.height(8.dp))
            }
            if (!modelReady) {
                Text(
                    if (modelError == null) "识别模型加载中…" else "识别模型未就绪：$modelError",
                    color = if (modelError == null) {
                        MaterialTheme.colorScheme.onSurface
                    } else {
                        MaterialTheme.colorScheme.error
                    }
                )
                if (modelError != null) {
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(onClick = onRetryModel) { Text("重试加载模型") }
                }
                Spacer(Modifier.height(8.dp))
            }
            Text("请将牌均匀铺开入镜，尽量避免重叠", style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = expectedText,
                onValueChange = onExpectedChange,
                label = { Text("期望张数") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                isError = expectedText.isNotEmpty() && expectedCount == null,
                modifier = Modifier.width(140.dp)
            )
            if (expectedText.isNotEmpty() && expectedCount == null) {
                Text(
                    "请输入 1–999 之间的张数",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall
                )
            }
            Spacer(Modifier.height(8.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(
                    onClick = {
                        if (busy || expectedCount == null) return@Button
                        busy = true
                        imageCapture.targetRotation = previewView?.display?.rotation
                            ?: Surface.ROTATION_0
                        val name = "MJ_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                            .format(Date()) + ".jpg"
                        val photoDir = context.getExternalFilesDir(null) ?: context.cacheDir
                        val photoFile = File(photoDir, name)
                        val output = ImageCapture.OutputFileOptions.Builder(photoFile).build()
                        imageCapture.takePicture(
                            output,
                            ContextCompat.getMainExecutor(context),
                            object : ImageCapture.OnImageSavedCallback {
                                override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                                    busy = false
                                    Log.i(
                                        TAG,
                                        "photo saved: ${photoFile.absolutePath}, bytes=${photoFile.length()}"
                                    )
                                    onPhotoTaken(photoFile)
                                }

                                override fun onError(exception: ImageCaptureException) {
                                    busy = false
                                    Log.e(
                                        TAG,
                                        "photo failed: ${photoFile.absolutePath}, exists=${photoFile.exists()}",
                                        exception
                                    )
                                    Toast.makeText(
                                        context,
                                        "拍照失败：${exception.message ?: "未知错误"}",
                                        Toast.LENGTH_SHORT
                                    ).show()
                                }
                            }
                        )
                    },
                    enabled = hasPermission && modelReady && !busy && expectedCount != null,
                    modifier = Modifier.weight(1f)
                ) {
                    Text(if (busy) "拍摄中…" else "拍照识别")
                }
                OutlinedButton(
                    onClick = { galleryLauncher.launch("image/*") },
                    enabled = modelReady && !busy && expectedCount != null,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("从相册选择")
                }
            }
            Spacer(Modifier.height(8.dp))
            OutlinedButton(
                onClick = onOpenHistory,
                enabled = !busy,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("历史记录")
            }
        }
    }
}

@Composable
fun ProcessingScreen(state: ScreenState.Processing) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator()
            Spacer(Modifier.height(16.dp))
            Text(
                if (state.total > 0) "正在识别：${state.done}/${state.total}"
                else "正在识别…"
            )
        }
    }
}

@Composable
fun ResultScreen(image: Bitmap, lines: List<String>, onRetake: () -> Unit) {
    DisposableEffect(image) {
        onDispose { if (!image.isRecycled) image.recycle() }
    }
    Column(Modifier.fillMaxSize()) {
        Image(
            bitmap = image.asImageBitmap(),
            contentDescription = "识别结果",
            contentScale = ContentScale.Fit,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        )
        Column(
            Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(16.dp)
        ) {
            lines.forEach {
                Text(it, style = MaterialTheme.typography.bodyLarge)
                Spacer(Modifier.height(6.dp))
            }
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = onRetake, modifier = Modifier.weight(1f)) {
                    Text("重新拍摄")
                }
                OutlinedButton(onClick = onRetake, modifier = Modifier.weight(1f)) {
                    Text("完成")
                }
            }
        }
    }
}

@Composable
private fun ErrorScreen(message: String, onRetake: () -> Unit, onRetryModel: () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(24.dp)
        ) {
            Text(message, color = MaterialTheme.colorScheme.error)
            Spacer(Modifier.height(16.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = onRetake) { Text("重新拍摄") }
                OutlinedButton(onClick = onRetryModel) { Text("重试模型") }
            }
        }
    }
}
