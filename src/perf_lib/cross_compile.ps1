
$root_path="D:\code\transsion\GearDVFS\src\perf_lib"

Remove-Item -Recurse "$root_path/build"

cmake -S $root_path `
-B "$root_path/build" `
-DCMAKE_TOOLCHAIN_FILE="D:/env_library/android_sdk/ndk/26.0.10792818/build/cmake/android.toolchain.cmake" `
-G "Ninja"

cmake --build "$root_path/build"