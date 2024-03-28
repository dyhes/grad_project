#include <iostream>
#include <thread>
#include <string>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "helper.h"
#include <cuda.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations (cutlass_tensorop_h16816gemm_128x128_32x4_nn_align8)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = float;//cutlass::half_t;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = float;//cutlass::half_t;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::RowMajor;                      // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = float;//cutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C/D matrices in units of elements (up to 16 bytes)

// Multiply-accumulate blocking/pipelining details
using ElementAccumulator  = float;//cutlass::half_t;                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm80;                      // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;           // Operator class tag
using ThreadblockShape    = cutlass::gemm::GemmShape<128, 128, 16>;   // Threadblock-level tile size (concept: GemmShape)
using WarpShape           = cutlass::gemm::GemmShape<64, 64, 16>;     // Warp-level tile size (concept: GemmShape)
using InstructionShape    = cutlass::gemm::GemmShape<16, 8, 8>;       // Instruction-level tile size (concept: GemmShape)
constexpr int NumStages   = 4;                                        // Number of global->shared pipeline stages used in the GEMM mainloop

// Epilogue output operator
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,               // Element type for C and D matrix operands
    AlignmentC,             // Memory access granularity of C and D matrix in units of elements
    ElementAccumulator,     // Element type from internal accumaccumulation
    ElementAccumulator>;    // Data type used to compute linear combination

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

// Classic data-parallel device GEMM implementation type
using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB>;

// StreamK device GEMM implementation type
using DeviceGemmStreamK = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, // <-- Only difference
    NumStages,
    AlignmentA,
    AlignmentB>;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(true)
  {}

};


/// Command line options parsing
struct Options
{
  std::string               command_name;
  bool                      help;
  cutlass::gemm::GemmCoord  problem_size;
  float                     alpha;
  float                     beta;
  int                       split_k_factor;
  int                       avail_sms;
  bool                      reference_check;
  int                       iterations;

  cutlass::HostTensor<ElementA, LayoutA> tensor_a;
  cutlass::HostTensor<ElementB, LayoutB> tensor_b;
  cutlass::HostTensor<ElementC, LayoutC> tensor_c;
  cutlass::HostTensor<ElementC, LayoutC> tensor_d;
  cutlass::HostTensor<ElementC, LayoutC> tensor_ref_d;

  Options(std::string command_name) :
    command_name(command_name),
    help(false),
    problem_size({2048, 2048, 2048}),
    alpha(1.0f),
    beta(0.0f),
    split_k_factor(1),
    avail_sms(-1),              // Number of device SMs to use is unlimited
    reference_check(true),
    iterations(10000)
  {}

  bool valid() const
  {
    return true;
  }

  void parse(int argc, char const **args)
  {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("split", split_k_factor);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const
  {
    out
      << "Performs a GEMM computation.\n"
      << "\n"
      << "Options:\n"
      << "\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --split=<int>               Split-K factor to emulate\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << command_name << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    return 2.0 * double(problem_size.product()) / double(1.0e9) / runtime_s;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Populates a DeviceGemmBasic::Arguments structure from the given commandline options
typename DeviceGemmBasic::Arguments args_from_options(
    const DeviceGemmBasic &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA, LayoutA> &tensor_a,
    cutlass::HostTensor<ElementB, LayoutB> &tensor_b,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_c,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_d)
{
  return typename DeviceGemmBasic::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
    options.problem_size,                     // problem_size
    options.split_k_factor,                   // batch count / splitk slices
    {                                         // epilogue parameters
      ElementAccumulator(options.alpha),
      ElementAccumulator(options.beta)
    },
    tensor_a.device_data(),                   // ptr_A
    tensor_b.device_data(),                   // ptr_B
    tensor_c.device_data(),                   // ptr_C
    tensor_d.device_data(),                   // ptr_D
    options.problem_size.mk().product(),      // batch_stride_A
    options.problem_size.nk().product(),      // batch_stride_B
    options.problem_size.mn().product(),      // batch_stride_C
    options.problem_size.mn().product(),      // batch_stride_D
    tensor_a.layout().stride(0),              // stride_a
    tensor_b.layout().stride(0),              // stride_b
    tensor_c.layout().stride(0),              // stride_c
    tensor_d.layout().stride(0));             // stride_d
}

/// Populates a DeviceGemmStreamK::Arguments structure from the given commandline options
typename DeviceGemmStreamK::Arguments args_from_options(
    const DeviceGemmStreamK &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA, LayoutA> &tensor_a,
    cutlass::HostTensor<ElementB, LayoutB> &tensor_b,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_c,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_d)
{
  return typename DeviceGemmStreamK::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
    options.problem_size,                     // problem_size
    options.split_k_factor,                   // batch count / splitk slices
    {                                         // epilogue parameters
      ElementAccumulator(options.alpha),
      ElementAccumulator(options.beta)
    },
    tensor_a.device_data(),                   // ptr_A
    tensor_b.device_data(),                   // ptr_B
    tensor_c.device_data(),                   // ptr_C
    tensor_d.device_data(),                   // ptr_D
    options.problem_size.mk().product(),      // batch_stride_A
    options.problem_size.nk().product(),      // batch_stride_B
    options.problem_size.mn().product(),      // batch_stride_C
    options.problem_size.mn().product(),      // batch_stride_D
    tensor_a.layout().stride(0),              // stride_a
    tensor_b.layout().stride(0),              // stride_b
    tensor_c.layout().stride(0),              // stride_c
    tensor_d.layout().stride(0),              // stride_d
    options.avail_sms);                       // avail_sms
}

 Result run(Options &options)
 {
  // Instantiate CUTLASS kernel depending on templates
  DeviceGemmStreamK device_gemm;
  Result result;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmStreamK
  auto arguments = args_from_options(device_gemm, options, options.tensor_a, options.tensor_b, options.tensor_c, options.tensor_c);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = DeviceGemmStreamK::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  CUTLASS_CHECK(device_gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));
  
  GpuTimer timer;
  timer.start();
  for (int iter = 0; iter < options.iterations; ++iter) {
    CUTLASS_CHECK(device_gemm());
  }
  timer.stop();
  // Compute average runtime and GFLOPs.
  float elapsed_ms = timer.elapsed_millis();
  result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
  result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);
  return result;
 }

float pos = 1.0f;
float neg = -1.0f;
int avail_sms = -1;

void strassen1(int m, int n, int k,
    float alpha,
     cutlass::TensorView<ElementA, LayoutA> &A, int ldA,
     cutlass::TensorView<ElementA, LayoutA> &B, int ldB,
    float beta,
     cutlass::TensorView<ElementA, LayoutA> &C, int ldC,
    float gamma,
     cutlass::TensorView<ElementA, LayoutA> &D
     ) {
      if (m <= 128 || n <= 128 || k <= 32) {
        DeviceGemmStreamK device_gemm;
        cutlass::TensorRef<ElementA, LayoutA> A_ref = A.ref();
        cutlass::TensorRef<ElementA, LayoutA> B_ref = B.ref();
        cutlass::TensorRef<ElementA, LayoutA> C_ref = C.ref();
        auto arguments = typename DeviceGemmStreamK::Arguments(
          cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
          {m, n, k},                     // problem_size
          1,                   // batch count / splitk slices
          {                                         // epilogue parameters
            ElementAccumulator(alpha),
            ElementAccumulator(beta)
          },
          A_ref.data(),                   // ptr_A
          B_ref.data(),                   // ptr_B
          C_ref.data(),                   // ptr_C
          C_ref.data(),                   // ptr_D
          m*k,      // batch_stride_A
          n*k,      // batch_stride_B
          m*n,      // batch_stride_C
          m*n,      // batch_stride_D
          A_ref.layout().stride(0),              // stride_a
          B_ref.layout().stride(0),              // stride_b
          C_ref.layout().stride(0),              // stride_c
          C_ref.layout().stride(0),              // stride_d
          avail_sms);                       // avail_sms
        size_t workspace_size = DeviceGemmStreamK::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));
        CUTLASS_CHECK(device_gemm());
        return;
      }
      cublasHandle_t handles[4];
      cudaStream_t streams[4];
      for(int i = 0; i < 4; i++) {
        cublasCreate(&handles[i]);
        cudaStreamCreate(&streams[i]);
        cublasSetStream(handles[i], streams[i]);
      }
      int subm = m / 2;
      int subn = n / 2;
      int subk = k / 2;
      cutlass::TensorView<ElementA, LayoutA> A0 = A.subview({subm, subk}, {0, 0});
      cutlass::TensorView<ElementA, LayoutA> A1 = A.subview({subm, subk}, {0, subk});
      cutlass::TensorView<ElementA, LayoutA> A2 = A.subview({subm, subk}, {subm, 0});
      cutlass::TensorView<ElementA, LayoutA> A3 = A.subview({subm, subk}, {subm, subk});
      cutlass::TensorView<ElementB, LayoutB> B0 = B.subview({subk, subn}, {0, 0});
      cutlass::TensorView<ElementB, LayoutB> B1 = B.subview({subk, subn}, {0, subn});
      cutlass::TensorView<ElementB, LayoutB> B2 = B.subview({subk, subn}, {subk, 0});
      cutlass::TensorView<ElementB, LayoutB> B3 = B.subview({subk, subn}, {subk, subn});
      cutlass::TensorView<ElementC, LayoutC> C0 = C.subview({subm, subn}, {0, 0});
      cutlass::TensorView<ElementC, LayoutC> C1 = C.subview({subm, subn}, {0, subn});
      cutlass::TensorView<ElementC, LayoutC> C2 = C.subview({subm, subn}, {subm, 0});
      cutlass::TensorView<ElementC, LayoutC> C3 = C.subview({subm, subn}, {subm, subn});
      cutlass::TensorRef<ElementA, LayoutA> A0_ref = A0.ref();
      cutlass::TensorRef<ElementA, LayoutA> A1_ref = A1.ref();
      cutlass::TensorRef<ElementA, LayoutA> A2_ref = A2.ref();
      cutlass::TensorRef<ElementA, LayoutA> A3_ref = A3.ref();
      cutlass::TensorRef<ElementB, LayoutB> B0_ref = B0.ref();
      cutlass::TensorRef<ElementB, LayoutB> B1_ref = B1.ref();
      cutlass::TensorRef<ElementB, LayoutB> B2_ref = B2.ref();
      cutlass::TensorRef<ElementB, LayoutB> B3_ref = B3.ref();
      cutlass::TensorRef<ElementC, LayoutC> C0_ref = C0.ref();
      cutlass::TensorRef<ElementC, LayoutC> C1_ref = C1.ref();
      cutlass::TensorRef<ElementC, LayoutC> C2_ref = C2.ref();
      cutlass::TensorRef<ElementC, LayoutC> C3_ref = C3.ref();

      cublasSgeam(handles[0], CUBLAS_OP_T, CUBLAS_OP_T, subm, subk, &pos, A0_ref.data(), subm, &pos, A1_ref.data(), subm, A1_ref.data(), subm);
      cublasSgeam(handles[1], CUBLAS_OP_T, CUBLAS_OP_T, subm, subk, &pos, A2_ref.data(), subm, &pos, A3_ref.data(), subm, A2_ref.data(), subm);
      cublasSgeam(handles[2], CUBLAS_OP_T, CUBLAS_OP_T, subk, subn, &pos, B1_ref.data(), subk, &neg, B3_ref.data(), subk, B1_ref.data(), subk);
      cublasSgeam(handles[3], CUBLAS_OP_T, CUBLAS_OP_T, subk, subn, &pos, B2_ref.data(), subk, &neg, B0_ref.data(), subk, B2_ref.data(), subk);
      cudaStreamSynchronize (streams[0]);
      thread t1(strassen1, subm, subk, subn, 1.0f, std::ref(A1), subm, std::ref(B3), subk, 1.0f, std::ref(C0), subm, 1.0f, std::ref(C1));
      cudaStreamSynchronize (streams[1]);
      thread t2(strassen1, subm, subk, subn, 1.0f, std::ref(A2), subm, std::ref(B0), subk, 1.0f, std::ref(C2), subm, -1.0f, std::ref(C3));
      cudaStreamSynchronize (streams[2]);
      cudaStreamSynchronize (streams[3]);
      t1.join();
      t2.join();
      cublasSgeam(handles[0], CUBLAS_OP_T, CUBLAS_OP_T, subk, subn, &pos, B0_ref.data(), subk, &pos, B3_ref.data(), subk, B0_ref.data(), subk);
      thread t3(strassen1, subm, subk, subn, 1, std::ref(A0), subm, std::ref(B1), subk, 1, std::ref(C1) , subm, 1, std::ref(C3));
      thread t4(strassen1, subm, subk, subn, 1, std::ref(A3), subm, std::ref(B2), subk, 1, std::ref(C0), subm, 1, std::ref(C2));
      t3.join();
      cublasSgeam(handles[1], CUBLAS_OP_T, CUBLAS_OP_T, subm, subk, &pos, A0_ref.data(), subm, &pos, A3_ref.data(), subm, A0_ref.data(), subm);
      cudaStreamSynchronize (streams[0]);
      cudaStreamSynchronize (streams[1]);
      t4.join();
      thread t5(strassen1, subm, subk, subn, 1, std::ref(A0), subm, std::ref(B0), subk, 1, std::ref(C0), subm, 1, std::ref(C3));
      cublasSgeam(handles[0], CUBLAS_OP_T, CUBLAS_OP_T, subm, subk, &pos, A2_ref.data(), subm, &neg, A0_ref.data(), subm, A2_ref.data(), subm);
      cublasSgeam(handles[1], CUBLAS_OP_T, CUBLAS_OP_T, subm, subk, &pos, A1_ref.data(), subm, &neg, A0_ref.data(), subm, A1_ref.data(), subm);
      cublasSgeam(handles[2], CUBLAS_OP_T, CUBLAS_OP_T, subk, subn, &pos, B1_ref.data(), subk, &pos, B0_ref.data(), subk, B1_ref.data(), subk);
      cublasSgeam(handles[3], CUBLAS_OP_T, CUBLAS_OP_T, subk, subn, &pos, B2_ref.data(), subk, &pos, B0_ref.data(), subk, B2_ref.data(), subk);
      cudaStreamSynchronize (streams[0]);
      cudaStreamSynchronize (streams[1]);
      cudaStreamSynchronize (streams[2]);
      cudaStreamSynchronize (streams[3]);
      t5.join();
      cublasSgeam(handles[0], CUBLAS_OP_T, CUBLAS_OP_T, subm, subk, &pos, A0_ref.data(), subm, &neg, A3_ref.data(), subm, A0_ref.data(), subm);
      cublasSgeam(handles[1], CUBLAS_OP_T, CUBLAS_OP_T, subk, subn, &pos, B0_ref.data(), subk, &neg, B3_ref.data(), subk, B0_ref.data(), subk); 
      thread t6(strassen1, subm, subk, subn, 1, std::ref(A2), subm, std::ref(B1), subk, 1, std::ref(C3), subm, 1, std::ref(C3));
      thread t7(strassen1, subm, subk, subn, 1, std::ref(A1), subm, std::ref(B2), subk, 1, std::ref(C0), subm, 1, std::ref(C0));
      cudaStreamSynchronize (streams[0]);
      cudaStreamSynchronize (streams[1]);
      t6.join();
      cublasSgeam(handles[0], CUBLAS_OP_T, CUBLAS_OP_T, subm, subk, &pos, A2_ref.data(), subm, &pos, A0_ref.data(), subm, A2_ref.data(), subm);
      cublasSgeam(handles[1], CUBLAS_OP_T, CUBLAS_OP_T, subk, subn, &pos, B1_ref.data(), subk, &neg, B0_ref.data(), subk, B1_ref.data(), subk); 
      t7.join();
      cublasSgeam(handles[2], CUBLAS_OP_T, CUBLAS_OP_T, subm, subk, &pos, A1_ref.data(), subm, &pos, A3_ref.data(), subm, A1_ref.data(), subm);
      cublasSgeam(handles[3], CUBLAS_OP_T, CUBLAS_OP_T, subk, subn, &pos, B2_ref.data(), subk, &neg, B3_ref.data(), subk, B2_ref.data(), subk); 
      cudaStreamSynchronize (streams[0]);
      cudaStreamSynchronize (streams[1]);
      cudaStreamSynchronize (streams[2]);
      cudaStreamSynchronize (streams[3]);
      for(int i = 0; i < 4; i++) {
        cublasDestroy(handles[i]);
        cudaStreamDestroy(streams[i]);
      }
      return;
}
 
 int test(Options options)
 {
   if (options.help) {
     options.print_usage(std::cout) << std::endl;
     return 0;
   }
 
   std::cout <<
     options.iterations << " timing iterations of " <<
     options.problem_size.m() << " x " <<
     options.problem_size.n() << " x " <<
     options.problem_size.k() << " matrix-matrix multiply" << std::endl;
 
   if (!options.valid()) {
     std::cerr << "Invalid problem." << std::endl;
     return -1;
   }
 
 
   //
   // Initialize GEMM datasets
   //
 
   // Initialize tensors using CUTLASS helper functions
   options.tensor_a.resize(options.problem_size.mk());       // <- Create matrix A with dimensions M x K
   options.tensor_b.resize(options.problem_size.kn());       // <- Create matrix B with dimensions K x N
   options.tensor_c.resize(options.problem_size.mn());       // <- Create matrix C with dimensions M x N

   // Fill matrix A on host with uniform-random data [-2, 2]
   cutlass::reference::host::TensorFillRandomUniform(
       options.tensor_a.host_view(),
       1,
       ElementA(2),
       ElementA(-2),
       0);
 
   // Fill matrix B on host with uniform-random data [-2, 2]
   cutlass::reference::host::TensorFillRandomUniform(
       options.tensor_b.host_view(),
       1,
       ElementB(2),
       ElementB(-2),
       0); 
 
   // Copy data from host to GPU
   options.tensor_a.sync_device();
   options.tensor_b.sync_device();
   GpuTimer timer;

   // CublasSgemm
   cutlass::reference::host::TensorFill(options.tensor_c.host_view());
   options.tensor_c.sync_device();
   cublasHandle_t handle;
   cublasCreate(&handle);
   timer.start();
   for (int i = 0; i < options.iterations; ++i) {
    cublasStatus_t cublas_stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
      options.problem_size.m(), options.problem_size.n(), options.problem_size.k(),
      &options.alpha,
      options.tensor_a.device_data(), options.problem_size.m(),
      options.tensor_b.device_data(), options.problem_size.k(),
      &options.beta,
      options.tensor_c.device_data(), options.problem_size.m());
      if (cublas_stat != CUBLAS_STATUS_SUCCESS) {
        cout << "cublassgemm failed with code: " << cublas_stat << endl;
        return -1;
      }
   }
   timer.stop();
   float elapsed_ms = timer.elapsed_millis();
   double cublas_time = double(elapsed_ms) / double(options.iterations);
   double culbas_gflops = options.gflops(cublas_time / 1000.0);
  printf("CUBLAS_GEMM:   [%6.1f]GFlop/s  (%6.4f)ms\n", culbas_gflops, cublas_time);
 
  //cutlass gemm
  cutlass::reference::host::TensorFill(options.tensor_c.host_view());
  options.tensor_c.sync_device();
  Result streamk_default  = run(options);
  printf("OPTIMIZED_GEMM:   [%6.1f]GFlop/s  (%6.4f)ms\n", streamk_default.gflops, streamk_default.avg_runtime_ms);

  //raw_strassen
  cutlass::reference::host::TensorFill(options.tensor_c.host_view());
  options.tensor_c.sync_device();


  //mixed_strassen 
  cutlass::reference::host::TensorFill(options.tensor_c.host_view());
  options.tensor_c.sync_device();

  timer
  timer.start();
  for (int iter = 0; iter < options.iterations; ++iter) {
    CUTLASS_CHECK(device_gemm());
  }
  timer.stop();
  // Compute average runtime and GFLOPs.
  float elapsed_ms = timer.elapsed_millis();
  result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
  result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);
  return result;

  cublasDestroy(handle);
  return 0;
 }

int main(int argc, char const **argv) {
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11))
  {
  std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;

  // Returning zero so this test passes on older Toolkits. Its actions are no-op.
  return 0;
  }
  
  // Current device must must have compute capability at least 80
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  if (!((props.major * 10 + props.minor) >= 80))
  {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;

    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
  
  // Parse commandline options
  #if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0

  Options options("Strassen matrix multiply");
  options.parse(argc, argv);
  test(options);
  return 0;

  #else

  std::cout << "Verification by comparison with cuBLAS is disabled, "
    "either because the CMake option CUTLASS_ENABLE_CUBLAS "
    "was explicitly set to OFF, or because CMake could not find cuBLAS. \n";

  #endif

}