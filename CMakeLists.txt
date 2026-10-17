cmake_minimum_required(VERSION 3.28)
project(embeddinggemma_kfd VERSION 0.1.0 LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) # No modules; preserve Nix wrapper include flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LIBKFD_BUILD_TESTS OFF CACHE BOOL "Do not run upstream fault/stress tests")
add_subdirectory(vendor/libkfd EXCLUDE_FROM_ALL)

add_executable(kfd-smoke src/kfd_smoke.cpp)
target_link_libraries(kfd-smoke PRIVATE libkfd::kfd)
target_compile_options(kfd-smoke PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter)

set(EG2_CODE_OBJECTS "")
foreach(arch gfx90c gfx1103)
  set(object "${CMAKE_CURRENT_BINARY_DIR}/eg2-${arch}.elf")
  add_custom_command(OUTPUT "${object}"
    COMMAND "${LIBKFD_GPU_COMPILER}" --target=amdgcn--amdhsa -mcpu=${arch}
      -nogpulib -nostdlibinc -std=c23 -O2 -fno-sanitize=all
      -resource-dir=${LIBKFD_GPU_RESOURCE_DIR}
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/eg2.c" -o "${object}"
    DEPENDS kernels/eg2.c kernels/abi.h VERBATIM)
  list(APPEND EG2_CODE_OBJECTS "${object}")
endforeach()
add_custom_target(eg2-kernels ALL DEPENDS ${EG2_CODE_OBJECTS})
install(TARGETS kfd-smoke RUNTIME DESTINATION bin)
install(FILES ${EG2_CODE_OBJECTS} DESTINATION share/embeddinggemma-kfd)

set(EI_SRC vendor/embeddinggemma.c/src)
add_library(eg2-core STATIC
  ${EI_SRC}/gguf.c ${EI_SRC}/model.c ${EI_SRC}/tokenizer.c ${EI_SRC}/quants.c
  ${EI_SRC}/kernels.c ${EI_SRC}/parallel.c ${EI_SRC}/engine.c ${EI_SRC}/gemma2.c
  ${EI_SRC}/gemma2_fast.c ${EI_SRC}/gemma2_fast_x86.c ${EI_SRC}/media.c
  ${EI_SRC}/response_cache.c ${EI_SRC}/inference_service.c src/gemma2_kfd.cpp)
target_include_directories(eg2-core PUBLIC ${EI_SRC} src)
target_compile_definitions(eg2-core PUBLIC EI_ENABLE_KFD=1)
target_compile_options(eg2-core PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(eg2-core PUBLIC libkfd::kfd m)
add_executable(embeddinggemma-kfd ${EI_SRC}/server.c ${EI_SRC}/float_format.c)
target_link_libraries(embeddinggemma-kfd PRIVATE eg2-core)
install(TARGETS embeddinggemma-kfd RUNTIME DESTINATION bin)
add_executable(kfd-kernel-test tests/kernels.cpp)
target_link_libraries(kfd-kernel-test PRIVATE libkfd::kfd)
install(TARGETS kfd-kernel-test RUNTIME DESTINATION bin)
add_executable(kfd-gemm-bench tests/gemm_bench.cpp)
target_link_libraries(kfd-gemm-bench PRIVATE libkfd::kfd)
install(TARGETS kfd-gemm-bench RUNTIME DESTINATION bin)
include(CTest)
add_executable(pm4-padding-test tests/pm4_padding.cpp)
target_link_libraries(pm4-padding-test PRIVATE libkfd::kfd)
add_test(NAME pm4-padding COMMAND pm4-padding-test)
add_executable(process-lock-test tests/process_lock.cpp)
add_test(NAME process-lock COMMAND process-lock-test)

set(EG2_LICENSE_DIR "share/licenses/embeddinggemma-kfd")
install(FILES LICENSE THIRD_PARTY_NOTICES.md DESTINATION ${EG2_LICENSE_DIR})
install(FILES vendor/libkfd/LICENSE DESTINATION ${EG2_LICENSE_DIR} RENAME libkfd-LICENSE)
install(FILES vendor/embeddinggemma.c/LICENSE DESTINATION ${EG2_LICENSE_DIR} RENAME embeddinggemma.c-LICENSE)
install(FILES vendor/PATCHES.md vendor/versions.json DESTINATION ${EG2_LICENSE_DIR})
