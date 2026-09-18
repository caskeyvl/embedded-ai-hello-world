# lab0_hello_world — sine-inference LED/UART lab

You will build a small neural network that approximates `y = sin(x)` with a neural network, using ExecuTorch on an STM32F407 (Cortex-M4). You will drive the on-board LED's PWM brightness from the model's output, and print `x`, `y`, and `angle` over UART every 10 steps.

Note: [ColoradoEmbeddedAI/simple_sine_led](https://github.com/ColoradoEmbeddedAI/simple_sine_led) has code for iterating over a range of values, calculating sine, and then updating the brightness of the LED with PWM.

What we're building here is effectively that, extended in two ways.
* Instead of using the sinf function to calculate the sine, we're performing inference using a model you train.  See first_et_project for some code on how to perform inference.  
* You will print out x, y, and angle over UART every 10 steps.  Below is code you should use to get the format matching.

```
            if (x % 10U == 0U) {
                char msg[64];
                char *p = msg;
                p = append_str(p, "x=");
                p = append_uint(p, x);
                p = append_str(p, " y=");
                p = append_float(p, y, 4);
                p = append_str(p, " angle=");
                p = append_float(p, angle, 4);
                p = append_str(p, "\r\n");
                *p = '\0';
                uart_puts(msg);
            }
```



This assignment assumes you've already installed ExecuTorch and gotten
[first_et_project](https://github.com/ColoradoEmbeddedAI/first_et_project)
working.

## What's provided here

This directory is your starting point, taken from `first_et_project`'s base
build. It already contains everything you should **not** need to touch:

```
lab0_hello_world/
├── CMakeLists.txt, CMakePresets.json, arm-none-eabi.cmake   (do not modify — see below)
├── startup_stm32f407.s, stm32f407.ld, stm32f407.h           (do not modify)
├── src/
│   └── syscalls.c                                            (do not modify)
├── platforms/
│   └── stm32f4_discovery_full.repl   # Renode platform description, for
│                                      # optional local/manual testing (see
│                                      # "Manual/local Renode testing" below)
├── models/     # empty — you add train_and_export.{py,ipynb} + sine_model.pte
└── lib/        # empty — you add libexecutorch_selected_kernels.a
```

You'll fill in the rest yourself, ending up with this full layout to submit:

```
<your_submission>/
├── CMakeLists.txt, CMakePresets.json, arm-none-eabi.cmake   (do not modify)
├── startup_stm32f407.s, stm32f407.ld, stm32f407.h           (do not modify)
├── models/
│   ├── train_and_export.py     # your training/export code (or train_and_export.ipynb — see below)
│   └── sine_model.pte          # the exported model
├── src/
│   ├── syscalls.c               (do not modify)
│   ├── main.cpp                 # your firmware
│   └── main_host.cpp            # your host test harness
├── lib/
│   └── libexecutorch_selected_kernels.a   # see "Building your kernels library" below
└── ops_manifest.json             # see "ops_manifest.json format" below
```

The autograding script only ever reads `models/train_and_export.{py,ipynb}`, `models/sine_model.pte`,
`src/main.cpp`, `src/main_host.cpp`, `lib/libexecutorch_selected_kernels.a`, and
`ops_manifest.json` from your submission — it copies its own known-good copies
of everything else (`CMakeLists.txt`, the toolchain file, startup/linker/syscalls)
into a scratch build directory and builds there. **Your own `CMakeLists.txt`
etc. are never executed by the grader** — they're here only so you can build
and test locally the same way you'll be graded. Don't submit `platforms/` —
it isn't part of the graded contract and the grader ignores it; it's only
here for your own local testing (see below).

**`models/train_and_export.py` is never executed by the autograding script either** — it's
only checked for existence and archived alongside your submission for human
reference. If you trained on Google Colab (or otherwise worked in a notebook)
instead of running a local script, submit `models/train_and_export.ipynb`
in its place — either is accepted, but submit exactly one of the two (not
both). Everything the grader actually checks (`models/sine_model.pte`, the
kernels library, `ops_manifest.json`) is the *output* of training, not the
training code itself, so the grader doesn't care how you produced it.

Do not change `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS`, the toolchain file, or the
linker script — grading always uses the grader's own copies of these, so any
local changes to them won't be reflected in your grade.

## Why a `lib/` directory and `ops_manifest.json`?

`EXECUTORCH_SELECT_OPS_LIST` (the flag that controls which ExecuTorch
operator kernels get compiled in) only affects one library:
`libexecutorch_selected_kernels.a`. Everything else ExecuTorch-related
(`libexecutorch.a`, `libexecutorch_core.a`, `libportable_kernels.a`, etc.) is
shared and identical for every submission, built once by course staff. Since
your model's op list depends on your architecture, **you build and submit
your own `libexecutorch_selected_kernels.a`** — the grader links it against
its shared base rather than rebuilding all of ExecuTorch per submission
(which would take 15–30 minutes each and isn't practical at scale).

## Toolchain requirements (must match exactly)

- ExecuTorch **release/1.2 (v1.2.0)** — the same commit/tag the shared base
  is built from
- The unmodified `arm-none-eabi.cmake` in this directory (`-mcpu=cortex-m4
  -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard`)

These aren't just style preferences: `lib/libexecutorch_selected_kernels.a`
gets linked against the grader's shared, centrally-built libraries at grade
time. Any drift in ExecuTorch commit or ABI flags risks a subtle mismatch
(differing struct layouts, calling conventions, etc.) between your library
and the shared ones — which may not fail cleanly at link time and could
instead surface as a Renode fault or wrong behavior. Cross-compiling on
macOS is fine as long as these things match; the output is Cortex-M4
object code and the host OS that produced it doesn't matter.

**`arm-none-eabi-gcc` version.** `13.x` (available via Homebrew on macOS) is
the known-good reference, but newer toolchains also work — this lab's
reference solution was rebuilt and graded **100/100** with `arm-none-eabi-gcc
15.3` from the current Homebrew `gcc-arm-embedded` cask. The portability
fixes a newer GCC needs are already carried in the `CMakeLists.txt` and
`src/syscalls.c` provided here, so you shouldn't need to change anything. If
you do hit an unexplained Renode fault or link mismatch, dropping back to
`arm-none-eabi-gcc` 13.x is the first thing to try.

## Building your kernels library

This creates a new, separate ExecuTorch build directory (e.g.
`~/executorch/executorch-build/cortex-m4-mine`) inside your ExecuTorch
checkout, distinct from the shared `cortex-m4`/`host` trees — configured
with *your* op list so it produces a `libexecutorch_selected_kernels.a`
matching your model, without touching the shared trees other submissions
rely on. (Not a CMake *preset* in the `CMakePresets.json` sense — just a
separate out-of-source build directory.)

1. Run `python3 models/train_and_export.py` (or run all cells of
   `models/train_and_export.ipynb` if you trained on Colab) and read the
   printed op list (it lists every op your exported graph uses, in
   `.out`-variant form).
2. Follow the instructions in README-installation.md in [ColoradoEmbeddedAI/first_et_project/README-installation.md](https://github.com/ColoradoEmbeddedAI/first_et_project) to build a local copy of executorch
3. Now, configure and build your own Cortex-M4 ExecuTorch tree with your op list from this project:
   ```bash
   cmake -S ~/executorch/executorch -B ~/executorch/executorch-build/cortex-m4-mine \
     -DCMAKE_TOOLCHAIN_FILE=~/executorch/executorch/cmake/toolchains/arm-none-eabi.cmake \
     -DEXECUTORCH_SELECT_OPS_LIST="aten::add.out,aten::mul.out,aten::tanh.out,<...your ops...>" \
     # ...(remaining flags as documented by first_et_project's setup guide)
   # Builds libexecutorch_selected_kernels.a inside the ExecuTorch tree above --
   # does NOT copy it into your submission; that's step 4 below.
   cmake --build ~/executorch/executorch-build/cortex-m4-mine --target executorch_selected_kernels -j$(nproc)
   ```
4. Copy the result into the lib directory in this project:
   ```bash
   cp ~/executorch/executorch-build/cortex-m4-mine/libexecutorch_selected_kernels.a lib/
   ```
5. Create/update `ops_manifest.json`'s `executorch_select_ops_list` to match
   the ops passed to `EXECUTORCH_SELECT_OPS_LIST` in step 3 — see the format
   below.

## `ops_manifest.json` format

Create this file yourself at the top level of your submission:

```json
{
  "executorch_select_ops_list": [
    "aten::add.out",
    "aten::mul.out",
    "aten::tanh.out",
    "dim_order_ops::_to_dim_order_copy.out"
  ]
}
```
(Use the real key name `executorch_select_ops_list` — the exact ops you passed
to `EXECUTORCH_SELECT_OPS_LIST` when building `lib/libexecutorch_selected_kernels.a`
above, `.out` variants only.)

The grader independently inspects the strings embedded in your submitted
`.a` file and checks they match this manifest exactly — if they don't
match, your submission fails category F1, and categories B–E (everything
that depends on the STM32 build) score zero, since the grader can't trust
an unverified binary. Keep this file in sync whenever you change your model.

## Building and testing locally before submitting

Run these from inside this directory (once `src/main.cpp`, `src/main_host.cpp`,
`models/`, `lib/`, and `ops_manifest.json` are filled in):

```bash
# 1. Train and export (needs the ExecuTorch venv active)
source ~/executorch/et-env/bin/activate
python3 models/train_and_export.py
# If you trained on Colab instead, run models/train_and_export.ipynb there
# and download the resulting sine_model.pte + the notebook itself -- keep
# whichever form (script or notebook) you actually used, not both.
# Note the printed op list -> write it into ops_manifest.json, build
# lib/libexecutorch_selected_kernels.a per "Building your kernels library" above.
deactivate

# 2. Host build + inference accuracy check (fast, no hardware/emulator needed)
# Assumes the shared ~/executorch/executorch-build/host tree already exists
# (built once, with EXECUTORCH_BUILD_KERNELS_PORTABLE=ON). The host target
# whole-archives portable_ops_lib (all ops, unconditionally), so it does NOT
# need rebuilding per your op list -- EXECUTORCH_SELECT_OPS_LIST only affects
# executorch_selected_kernels (used by the stm32 target below).
cmake --preset host && cmake --build --preset host
./build/host/inference_host models/sine_model.pte     # should print PASS

# 3. STM32 build
# Assumes you've already built lib/libexecutorch_selected_kernels.a with your
# op list (see "Building your kernels library" above) -- this target links
# against that library, which IS op-list-specific.
cmake --preset stm32 && cmake --build --preset stm32
arm-none-eabi-size build/stm32/firmware.elf            # check it fits flash/RAM

# 4. Optional: flash to real hardware and observe manually
cmake --build build/stm32 --target flash
# The Discovery board has no onboard USB-UART bridge -- connect a 3.3V USB-UART
# adapter to the P2 header (USART2: PA2=TX) and open a terminal:
picocom -b 115200 --imap lfcrlf /dev/ttyUSB0
# adjust /dev/ttyUSB0 to your adapter's device node; exit with Ctrl-A then Ctrl-X
#
# macOS equivalent (picocom installs via Homebrew too):
picocom -b 115200 --imap lfcrlf /dev/cu.usbserial-XXXX
# find your adapter's actual device node with `ls /dev/cu.usbserial-*` --
# use cu. (not tty.) so picocom doesn't block waiting for carrier detect
#
# What to observe:
#  - The green user LED (PD12) should fade smoothly in and out, tracking the
#    inferred sine wave's PWM duty cycle.
#  - The terminal should print a line every 10 steps, e.g.:
#      x=0 y=0.0012 angle=0.0000
#      x=10 y=0.5432 angle=0.1745
#    x wraps 0..359 (degrees), y is the model's inferred sin(angle), and the
#    LED brightness should visibly track y as x sweeps through a full cycle.
```

### Manual/local Renode testing

`platforms/stm32f4_discovery_full.repl` describes the same board Renode uses
when grading you — it lets you load your own `firmware.elf` into Renode
and poke at it (watch the LED GPIO, bridge USART2 to a terminal, etc.)
without real hardware or waiting for a graded run. A minimal script:

```
mach create
machine LoadPlatformDescription @platforms/stm32f4_discovery_full.repl
sysbus LoadELF @build/stm32/firmware.elf
emulation CreateServerSocketTerminal 3456 "uart_term" false
connector Connect sysbus.usart2 uart_term
start
```

Save that as e.g. `demo.resc` and run `renode demo.resc`, then connect to
the terminal socket (e.g. `nc localhost 3456`) to watch UART output live.
This is entirely optional and separate from grading — the grader runs its
own Robot Framework tests against your build, not this script.

## Grading rubric

100 points total, split into these categories (each `grade.py` run's JSON
output includes one entry per category below, with a `detail` field that
explains what went wrong and how to fix it if you didn't get full credit):

| ID | What it checks | Points |
|----|-----------------|-------:|
| SHAPE | Submission has all required files (see "What's provided here" / final layout above), including exactly one of `train_and_export.py`/`train_and_export.ipynb` | gate, not points — passing earns nothing on its own; failing scores the whole submission 0/100 and skips every category below |
| F1 | `ops_manifest.json` matches the ops actually embedded in your submitted `lib/libexecutorch_selected_kernels.a` | 5 |
| A1 | Host build (`cmake --preset host`) succeeds | 5 |
| A2 | Host inference mean error vs. `sinf()` is small | 15 |
| B1 | STM32 build (`cmake --preset stm32`) succeeds within the flash/RAM budget | 5 |
| B2 | Firmware boots and runs fault-free under Renode (CFSR == 0) | 5 |
| C1 | Average PWM duty cycle over a full sweep is ~50% | 10 |
| C2 | PWM duty cycle reaches near its ~100% peak | 7.5 |
| C3 | PWM duty cycle reaches near its ~0% trough | 7.5 |
| D1 | UART output has the expected number of well-formed `x=... y=... angle=...` lines | 10 |
| D2 | The sequence of `x` values printed is `0, 10, 20, ...` as expected | 5 |
| D3 | Printed `angle` values match `x` converted to radians | 5 |
| D4 | Printed `y` values match `sin(angle)` closely | 15 |
| E1 | Steps land close to the expected ~100ms cadence | 5 |

Categories F1 and B1–E1 all depend on a successful, verified STM32 build —
if F1 fails (manifest/library mismatch) or B1 fails (STM32 build failure),
every category that depends on it is scored 0 with a `detail` explaining
which upstream category to fix first. A1/A2 (the host build/accuracy check)
are independent of your STM32 build and kernels library, so they're always
evaluated on their own merits.

