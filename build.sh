#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -d "$ROOT_DIR/tools/jdk-17/Contents/Home" ]; then
    export JAVA_HOME="$ROOT_DIR/tools/jdk-17/Contents/Home"
fi
if [ -d "$ROOT_DIR/android-sdk" ]; then
    export ANDROID_SDK_ROOT="$ROOT_DIR/android-sdk"
    export ANDROID_HOME="$ROOT_DIR/android-sdk"
fi

if [ -x "$ROOT_DIR/tools/gradle-8.9/bin/gradle" ]; then
    exec "$ROOT_DIR/tools/gradle-8.9/bin/gradle" "$@"
fi
exec "$ROOT_DIR/gradlew" "$@"
