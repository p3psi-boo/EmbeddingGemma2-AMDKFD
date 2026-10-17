target_sources(kfd PRIVATE
  device.cpp
  event.cpp
  context.cpp
  loader.cpp
  memory.cpp
  queue.cpp
  signal.cpp
  topology.cpp
  trap_handler.c

  detail/scratch.cpp
  detail/elf.cpp
  detail/mapped_region.cpp
  detail/pool_allocator.cpp
)

# Build per-architecture AMDGCN trap handler executables and pass HAS_<arch>
# defines so trap_handler.c knows which #embed blobs to pull in.
set(trap_targets "")
set(trap_arch_defines "")
foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
    set(target trap_handler_${arch}.bin)
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${target}"
      COMMAND "${LIBKFD_GPU_COMPILER}" ${LIBKFD_GPU_FLAGS} -mcpu=${arch}
        -Wl,--entry=trap_entry "${CMAKE_CURRENT_SOURCE_DIR}/device/trap_handler.S"
        -o "${CMAKE_CURRENT_BINARY_DIR}/${target}"
      DEPENDS device/trap_handler.S VERBATIM)
    add_custom_target(build_${target} DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${target}")
    list(APPEND trap_targets build_${target})

    string(REPLACE "-" "_" arch_id "${arch}")
    string(TOUPPER "${arch_id}" arch_id)
    list(APPEND trap_arch_defines "HAS_${arch_id}")
endforeach()

target_compile_options(kfd PRIVATE --embed-dir=${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(kfd PRIVATE ${trap_arch_defines})
add_dependencies(kfd ${trap_targets})
