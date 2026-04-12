cmake_minimum_required(VERSION 3.16)
project(kompass_benchmark_mojo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(kompass_benchmark_mojo benchmark_runner_mojo.cpp)
target_include_directories(kompass_benchmark_mojo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(kompass_benchmark_mojo PRIVATE -Wall -Wextra -O3)

# Link against libkompass_mojo.so built by scripts/build_shared_lib.sh.
target_link_directories(kompass_benchmark_mojo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../build
)
target_link_libraries(kompass_benchmark_mojo PRIVATE kompass_mojo)

# Bake the shared-library search paths into the binary via RPATH so it runs
# without needing LD_LIBRARY_PATH. libkompass_mojo.so lives next to the
# binary and pulls in the Mojo runtime libraries from the pixi environment,
# so we add both.
set(PIXI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.pixi/envs/default/lib"
    CACHE PATH "Mojo runtime library directory (pixi env)")
set_target_properties(kompass_benchmark_mojo PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../build;${PIXI_LIB_DIR}"
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../build;${PIXI_LIB_DIR}"
)

# Opt-in hardware power monitoring. Enable on supported hardware
option(ENABLE_POWER_MONITOR "Enable hardware power monitoring during benchmarks" OFF)
if(ENABLE_POWER_MONITOR)
    target_compile_definitions(kompass_benchmark_mojo PRIVATE ENABLE_POWER_MONITOR=1)
    find_package(Threads REQUIRED)
    target_link_libraries(kompass_benchmark_mojo PRIVATE Threads::Threads)
endif()
