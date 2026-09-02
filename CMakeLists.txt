cmake_minimum_required(VERSION 3.20)
project(sine_infer_led CXX C ASM)

# ------------------------------------------------------------------ #
#  Validation                                                          #
# ------------------------------------------------------------------ #
if(NOT DEFINED TARGET_PLATFORM)
    message(FATAL_ERROR
        "TARGET_PLATFORM must be set. Use a CMake preset:\n"
        "  cmake --preset host   (Linux debug)\n"
        "  cmake --preset stm32  (STM32F407 bare-metal)")
endif()

if(NOT DEFINED EXECUTORCH_BUILD_DIR)
    message(FATAL_ERROR "EXECUTORCH_BUILD_DIR must be set (via preset).")
endif()

# ExecuTorch headers come from the source tree, not the build tree.
set(EXECUTORCH_SRC_DIR "$ENV{HOME}/executorch")

# Where the (op-list-dependent) executorch_selected_kernels library lives.
# Defaults to this project's own lib/ so the reference solution builds
# standalone; grade.py overrides this per submission when building from a
# scratch/skeleton dir, since each submission brings its own copy of this
# one library (see lab0_hello_world_soln/README.md and
# lab0_hello_world_test/README.md for why).
if(NOT DEFINED SUBMISSION_LIB_DIR)
    set(SUBMISSION_LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
endif()

message(STATUS "TARGET_PLATFORM:      ${TARGET_PLATFORM}")
message(STATUS "EXECUTORCH_BUILD_DIR: ${EXECUTORCH_BUILD_DIR}")
message(STATUS "EXECUTORCH_SRC_DIR:   ${EXECUTORCH_SRC_DIR}")
message(STATUS "SUBMISSION_LIB_DIR:   ${SUBMISSION_LIB_DIR}")

set(ET_LIB_DIR ${EXECUTORCH_BUILD_DIR})

# ------------------------------------------------------------------ #
#  Imported ExecuTorch libraries                                       #
#                                                                      #
#  All of these except executorch_selected_kernels are op-list-        #
#  independent and come from one shared, centrally-built tree          #
#  (EXECUTORCH_BUILD_DIR). executorch_selected_kernels is the one       #
#  library EXECUTORCH_SELECT_OPS_LIST actually affects, so it alone     #
#  comes from SUBMISSION_LIB_DIR (this repo's own lib/ by default).     #
# ------------------------------------------------------------------ #

add_library(executorch_core STATIC IMPORTED)
set_target_properties(executorch_core PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch_core.a)

add_library(executorch STATIC IMPORTED)
set_target_properties(executorch PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch.a)

# Only needed for the stm32 target -- the host target whole-archives
# portable_ops_lib directly instead (see host branch below), since it has
# no flash budget and thus no reason to depend on a per-submission library.
if(TARGET_PLATFORM STREQUAL "stm32")
    add_library(executorch_selected_kernels STATIC IMPORTED)
    set_target_properties(executorch_selected_kernels PROPERTIES
        IMPORTED_LOCATION ${SUBMISSION_LIB_DIR}/libexecutorch_selected_kernels.a)
endif()

add_library(portable_kernels STATIC IMPORTED)
set_target_properties(portable_kernels PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/kernels/portable/libportable_kernels.a)

# portable_ops_lib holds the *registration* glue (static initializers that
# register every portable kernel's op name into the operator registry) for
# ALL portable ops, unconditionally -- unlike executorch_selected_kernels,
# it doesn't need EXECUTORCH_SELECT_OPS_LIST. Only used by the host target
# (see CLAUDE.md: whole-archiving it on the flash-constrained stm32 target
# pulls in every op and blows the 1MB budget; host has no such budget).
add_library(portable_ops_lib STATIC IMPORTED)
set_target_properties(portable_ops_lib PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/kernels/portable/libportable_ops_lib.a)

add_library(extension_data_loader STATIC IMPORTED)
set_target_properties(extension_data_loader PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/extension/data_loader/libextension_data_loader.a)

add_library(flatccrt STATIC IMPORTED)
set_target_properties(flatccrt PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/third-party/flatcc_ep/lib/libflatccrt.a)

set(ET_INCLUDES
    ${EXECUTORCH_SRC_DIR}
    ${EXECUTORCH_SRC_DIR}/executorch/third-party/flatbuffers/include
    ${ET_LIB_DIR}/third-party/flatcc_ep/include
    $ENV{HOME}/executorch/et-env/lib/python3.12/site-packages/torch/include
)

# ------------------------------------------------------------------ #
#  Host (Linux) build                                                  #
# ------------------------------------------------------------------ #
if(TARGET_PLATFORM STREQUAL "host")

    add_executable(inference_host
        src/main_host.cpp
    )

    # -std=c++17: ExecuTorch headers need it (std::variant, is_same_v). Linux
    # GCC defaults to gnu++17; Apple Clang defaults to C++14, so pass it
    # explicitly. (Matches first_et_project/CMakeLists.txt.)
    target_compile_options(inference_host PRIVATE
        -std=c++17 -Wall -Wextra -g -Og
    )

    target_include_directories(inference_host PRIVATE
        ${ET_INCLUDES}
    )

    # Whole-archive portable_ops_lib (registers every portable op
    # unconditionally, regardless of EXECUTORCH_SELECT_OPS_LIST -- see
    # README.md); portable_kernels (the implementations it registers)
    # links normally, pulled in by symbol reference from the registration
    # table. No flash budget on host, so this sidesteps the per-submission
    # op-list problem entirely for the host build.
    # GNU ld spells this --whole-archive/--no-whole-archive around the lib;
    # Apple's ld64 (macOS) has no such flag and instead takes -force_load
    # <archive> (per archive, no closing flag). (Matches first_et_project.)
    if(APPLE)
        target_link_libraries(inference_host PRIVATE
            -Wl,-force_load,$<TARGET_FILE:portable_ops_lib>
            portable_kernels
            executorch
            executorch_core
            extension_data_loader
            flatccrt
        )
    else()
        target_link_libraries(inference_host PRIVATE
            -Wl,--whole-archive
            portable_ops_lib
            -Wl,--no-whole-archive
            portable_kernels
            executorch
            executorch_core
            extension_data_loader
            flatccrt
        )
    endif()

# ------------------------------------------------------------------ #
#  STM32F407 (bare-metal Cortex-M4) build                             #
# ------------------------------------------------------------------ #
elseif(TARGET_PLATFORM STREQUAL "stm32")

    set(CPU_FLAGS
        "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard"
    )

    set(CMAKE_C_FLAGS
        "${CPU_FLAGS} -Os -ffunction-sections -fdata-sections -mno-unaligned-access -DET_LOG_ENABLED=0"
        CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS
        "${CPU_FLAGS} -Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -std=c++17 -mno-unaligned-access -DET_LOG_ENABLED=0"
        CACHE STRING "" FORCE)

    # Convert .pte file to a C array header at configure time.
    # chdir into models/ so the generated variable name is "sine_model_pte"
    # rather than a mangled full-path name.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/models
            xxd -i sine_model.pte
        OUTPUT_VARIABLE XXD_OUTPUT
        RESULT_VARIABLE XXD_RESULT
    )
    if(NOT XXD_RESULT EQUAL 0)
        message(FATAL_ERROR
            "xxd failed. Is models/sine_model.pte present?\n"
            "Run: python3 models/train_and_export.py")
    endif()

    # ExecuTorch's FlatBuffer parser requires the program data pointer to be
    # aligned to alignof(std::max_align_t); the default char[] has no such
    # guarantee, so add __attribute__((aligned(16))) and const.
    string(REPLACE
        "unsigned char sine_model_pte[]"
        "__attribute__((aligned(16))) const unsigned char sine_model_pte[]"
        XXD_OUTPUT "${XXD_OUTPUT}")
    string(REPLACE
        "unsigned int sine_model_pte_len"
        "const unsigned int sine_model_pte_len"
        XXD_OUTPUT "${XXD_OUTPUT}")

    file(WRITE ${CMAKE_BINARY_DIR}/sine_model_pte.h "${XXD_OUTPUT}")
    message(STATUS "Generated sine_model_pte.h in ${CMAKE_BINARY_DIR}")

    add_executable(firmware.elf
        startup_stm32f407.s
        src/syscalls.c
        src/main.cpp
    )

    target_compile_options(firmware.elf PRIVATE -g -Og)

    target_include_directories(firmware.elf PRIVATE
        ${ET_INCLUDES}
        ${CMAKE_SOURCE_DIR}      # for stm32f407.h
        ${CMAKE_BINARY_DIR}      # for generated sine_model_pte.h
    )

    # See host target comments for --whole-archive rationale (op registration
    # happens via static initializers with no directly-referenced symbols).
    target_link_libraries(firmware.elf PRIVATE
        -Wl,--whole-archive
        executorch_selected_kernels
        -Wl,--no-whole-archive
        portable_kernels
        executorch
        executorch_core
        extension_data_loader
        flatccrt
    )

    # Linked via target_link_libraries (not embedded in target_link_options)
    # so they land after the object files on the link line.
    target_link_libraries(firmware.elf PRIVATE c m gcc)

    # -nostartfiles excludes crti.o/crtn.o; _init/_fini stubs are provided
    # in startup_stm32f407.s instead.
    target_link_options(firmware.elf PRIVATE
        -nostartfiles
        -specs=nano.specs
        -T${CMAKE_SOURCE_DIR}/stm32f407.ld
        -Wl,--gc-sections
        -Wl,-Map=${CMAKE_BINARY_DIR}/firmware.map
    )

    add_custom_command(TARGET firmware.elf POST_BUILD
        COMMAND arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
        COMMENT "Generating firmware.bin"
    )
    add_custom_command(TARGET firmware.elf POST_BUILD
        COMMAND arm-none-eabi-size firmware.elf
        COMMENT "Firmware size:"
    )

    add_custom_target(flash
        COMMAND openocd
            -f interface/stlink.cfg
            -f target/stm32f4x.cfg
            -c "transport select hla_swd"
            -c "program ${CMAKE_BINARY_DIR}/firmware.bin 0x08000000 verify reset exit"
        DEPENDS firmware.elf
        COMMENT "Flashing firmware via OpenOCD (chip runs after)"
        USES_TERMINAL
    )

    add_custom_target(debug
        COMMAND openocd
            -f interface/stlink.cfg
            -f target/stm32f4x.cfg
            -c "transport select hla_swd"
        DEPENDS firmware.elf
        COMMENT "Starting OpenOCD GDB server on :3333 (Ctrl-C to stop)"
        USES_TERMINAL
    )

    add_custom_target(disasm
        COMMAND arm-none-eabi-objdump -d -S ${CMAKE_BINARY_DIR}/firmware.elf
        DEPENDS firmware.elf
        COMMENT "Disassembly:"
        USES_TERMINAL
    )

else()
    message(FATAL_ERROR
        "Unknown TARGET_PLATFORM: '${TARGET_PLATFORM}'. "
        "Must be 'host' or 'stm32'.")
endif()
