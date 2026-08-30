#!/bin/sh
# 真机全量回归：对每张照片 force-stop -> 冷启动 debug 入口 -> 轮询等待结果。
# 用法: sh training/user/run_device_regression.sh "<配置标签>" 照片1.jpg [照片2.jpg ...]
# 结果追加到 training/user/device_regression/<标签>.csv，缩略图存同目录 <标签>/。
set -u
cd "$(dirname "$0")/../.."

LABEL="${1:?usage: $0 <label> photos...}"
shift
DEV="${REG_DEV:-192.168.1.233:41627}"
PKG=com.example.majiang.debug
ADB="android-sdk/platform-tools/adb -s $DEV"
OUT="training/user/device_regression"
mkdir -p "$OUT/$LABEL"

$ADB connect "$DEV" >/dev/null 2>&1
sleep 1
STATE=$($ADB get-state 2>&1)
if [ "$STATE" != "device" ]; then
  echo "[$(date '+%T')] ERROR: 设备不可用 ($STATE)"
  exit 1
fi
# 保活
$ADB shell settings put system screen_off_timeout 1800000 >/dev/null
$ADB shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1

CSV="$OUT/$LABEL.csv"
[ -f "$CSV" ] || echo "photo,complete_ts,count,seconds" >> "$CSV"

for photo in "$@"; do
  name=$(basename "$photo" .jpg)
  $ADB push "$photo" /data/local/tmp/reg_in.jpg >/dev/null
  $ADB shell "run-as $PKG mkdir -p files" >/dev/null
  $ADB shell "run-as $PKG cp /data/local/tmp/reg_in.jpg files/reg.jpg" >/dev/null
  $ADB logcat -c
  $ADB shell am force-stop $PKG
  $ADB shell am start -n $PKG/com.example.majiang.MainActivity \
    --es debug_image /data/user/0/$PKG/files/reg.jpg >/dev/null

  result=""
  for i in $(seq 1 60); do
    sleep 5
    line=$($ADB logcat -d 2>/dev/null | grep "analysis complete" | tail -1)
    if [ -n "$line" ]; then
      result="$line"
      break
    fi
    err=$($ADB logcat -d 2>/dev/null | grep "analysis failed" | tail -1)
    [ -n "$err" ] && { result="$err"; break; }
  done

  ts=$(echo "$result" | awk '{print $2}')
  count=$(echo "$result" | sed -n 's/.*detections=\([0-9]*\).*/\1/p')
  started=$($ADB logcat -d 2>/dev/null | grep "analysis started" | tail -1)
  secs=$(python3 -c "
from datetime import datetime
import sys
def ts(l): return datetime.strptime(l.split()[1], '%H:%M:%S.%f')
lines = [l for l in sys.argv[1:] if 'analysis started' in l or 'analysis complete' in l]
print(f'{(ts(lines[-1]) - ts(lines[0])).total_seconds():.1f}')
" "$started" "$result" 2>/dev/null || echo NA)
  echo "[$(date '+%T')] $name -> count=${count:-NA} secs=${secs:-NA}"
  echo "$name,$ts,${count:-NA},${secs:-NA}" >> "$CSV"
  # 拉最新缩略图作为该照片的留档
  newest=$($ADB shell "run-as $PKG ls files/history/img/" | tr -d '\r' | sort | tail -1)
  [ -n "$newest" ] && $ADB shell "run-as $PKG cat files/history/img/$newest" > "$OUT/$LABEL/$name.jpg" 2>/dev/null
done

echo "[$(date '+%T')] 套件完成：$LABEL"
