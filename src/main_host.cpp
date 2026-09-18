/*
 * main_host.cpp — Linux host inference tester for the sine model
 *
 * Loads models/sine_model.pte, runs it across a sweep of angles, and
 * checks the model's output against sinf() to sanity-check the model +
 * ExecuTorch runtime *before* touching the STM32/Renode. No hardware
 * involved here at all — this is graded independently (rubric A1: builds,
 * A2: mean error is small).
 *
 * Usage:
 *   ./build/host/inference_host models/sine_model.pte
 */

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/executor/method.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/portable_type/tensor_impl.h"
#include "executorch/runtime/core/hierarchical_allocator.h"
#include "executorch/runtime/platform/runtime.h"
#include "executorch/extension/data_loader/file_data_loader.h"

using namespace executorch::runtime;
using namespace executorch::extension;
using namespace executorch::runtime::etensor;

static uint8_t method_pool[64 * 1024];
static uint8_t planned_pool[64 * 1024];

static constexpr float kPi = 3.14159265358979323846f;

static constexpr int kNumSteps = 360;
static constexpr float kMaxMeanError = 0.05f;

int main(int argc, char** argv) {
    const char* pte_path = (argc > 1) ? argv[1] : "models/sine_model.pte";

    runtime_init();

    // Host uses FileDataLoader — reads directly from disk (main.cpp instead
    // uses BufferDataLoader over a flash-embedded C array).
    Result<FileDataLoader> loader_result = FileDataLoader::from(pte_path);
    if (!loader_result.ok()) {
        fprintf(stderr, "ERROR: Could not open model: %s\n", pte_path);
        return 1;
    }
    FileDataLoader loader = std::move(loader_result.get());

    Result<Program> program_result = Program::load(&loader);
    if (!program_result.ok()) {
        fprintf(stderr, "ERROR: Program::load failed (code %d)\n",
                (int)program_result.error());
        return 1;
    }
    Program program = std::move(program_result.get());

    MemoryAllocator method_allocator(sizeof(method_pool), method_pool);
    Span<uint8_t> planned_span(planned_pool, sizeof(planned_pool));
    HierarchicalAllocator planned_allocator({&planned_span, 1});
    MemoryManager memory_manager(&method_allocator, &planned_allocator);

    Result<Method> method_result = program.load_method("forward", &memory_manager);
    if (!method_result.ok()) {
        fprintf(stderr, "ERROR: load_method failed (code %d)\n",
                (int)method_result.error());
        return 1;
    }
    Method method = std::move(method_result.get());

    // Input tensor: shape [1, 1] float32 -- one scalar angle in, one scalar
    // sine value out. Matches train_and_export.py's example_input =
    // torch.randn(1, 1). input_data is `static` so its address is stable
    // across the whole run, same reasoning as main.cpp.
    static float input_data[1] = {0.0f};
    static TensorImpl::SizesType    sizes[2]     = {1, 1};
    static TensorImpl::DimOrderType dim_order[2] = {0, 1};
    static TensorImpl::StridesType  strides[2]   = {1, 1};

    TensorImpl input_impl(
        ScalarType::Float,
        /*dim=*/2,
        sizes,
        input_data,
        dim_order,
        strides
    );
    Tensor input_tensor(&input_impl);

    // Sweep a full cycle, re-running the same loaded Method each time.
    // set_input()/execute() are cheap to call repeatedly on one loaded
    // Method -- no need to reload the program or method per test point.
    double sum_abs_error = 0.0;
    float  max_abs_error = 0.0f;

    for (int i = 0; i < kNumSteps; i++) {
        float angle = i * (2.0f * kPi / kNumSteps);
        input_data[0] = angle;

        Error set_err = method.set_input(EValue(input_tensor), 0);
        if (set_err != Error::Ok) {
            fprintf(stderr, "ERROR: set_input failed at i=%d\n", i);
            return 1;
        }

        Error exec_err = method.execute();
        if (exec_err != Error::Ok) {
            fprintf(stderr, "ERROR: execute() failed at i=%d\n", i);
            return 1;
        }

        EValue output = method.get_output(0);
        Tensor out_tensor = output.toTensor();
        float y = out_tensor.const_data_ptr<float>()[0];

        float expected = sinf(angle);
        float err = fabsf(y - expected);

        sum_abs_error += err;
        if (err > max_abs_error) max_abs_error = err;

        // Print a sample every 60 steps (6 lines total) so a human skimming
        // output can eyeball the model tracking sinf(), without flooding
        // the terminal with all 360 lines.
        if (i % 60 == 0) {
            printf("  angle=%7.4f  y=%8.5f  sinf=%8.5f  err=%.5f\n",
                   angle, y, expected, err);
        }
    }

    float mean_abs_error = (float)(sum_abs_error / kNumSteps);
    printf("\nMean abs error: %.5f (over %d points)\n", mean_abs_error, kNumSteps);
    printf("Max  abs error: %.5f\n", max_abs_error);

    bool correct = mean_abs_error < kMaxMeanError;
    printf("%s\n", correct ? "PASS" : "FAIL");
    return correct ? 0 : 1;
}
