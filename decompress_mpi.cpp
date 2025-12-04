#include <adios2.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <numeric>
#include <algorithm>

namespace fs = std::filesystem;

size_t getPathSize(const std::string& pathStr) {
    fs::path path(pathStr);
    if (!fs::exists(path)) return 0;
    if (fs::is_regular_file(path)) return fs::file_size(path);
    size_t size = 0;
    if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (fs::is_regular_file(entry.status())) size += fs::file_size(entry);
        }
    }
    return size;
}

template<typename T>
void analyze_and_write_step(
    adios2::Engine& readerOrig,
    adios2::Engine& readerComp,
    adios2::Engine& writerDecomp, // 쓰기 엔진
    adios2::IO& ioOrig,
    adios2::IO& ioComp,
    adios2::IO& ioWrite,          // 쓰기 IO
    const std::string& varName,
    size_t decompDim,
    int rank,
    int size
) {
    auto varOrig = ioOrig.InquireVariable<T>(varName);
    auto varComp = ioComp.InquireVariable<T>(varName);

    if (!varOrig || !varComp) return;

    auto shape = varOrig.Shape();
    size_t ndims = shape.size();
    if (decompDim >= ndims) return;

    std::vector<size_t> start(ndims, 0);
    std::vector<size_t> count = shape;
    count[decompDim] /= size;
    start[decompDim] = rank * count[decompDim];
    if (rank == size - 1) count[decompDim] = shape[decompDim] - start[decompDim];

    varOrig.SetSelection({start, count});
    varComp.SetSelection({start, count});

    size_t localSize = 1;
    for (auto c : count) localSize *= c;
    std::vector<T> dataOrig(localSize);
    std::vector<T> dataComp(localSize);

    readerOrig.Get(varOrig, dataOrig.data(), adios2::Mode::Sync);
    readerComp.Get(varComp, dataComp.data(), adios2::Mode::Sync);

    auto varOut = ioWrite.InquireVariable<T>(varName);
    if (!varOut) {
        varOut = ioWrite.DefineVariable<T>(varName, shape, start, count, adios2::ConstantDims);
    }
    
    if (varOut) {
        writerDecomp.Put(varOut, dataComp.data(), adios2::Mode::Sync);
    }
    // ---------------------------------------------------------

    double localSumSqErr = 0.0;
    double localSumSqOrig = 0.0;
    double localMaxErr = 0.0;
    double localMinVal = std::numeric_limits<double>::max();
    double localMaxVal = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < localSize; ++i) {
        double o = static_cast<double>(dataOrig[i]);
        double c = static_cast<double>(dataComp[i]);
        double err = std::abs(o - c);
        
        localSumSqErr += err * err;
        localSumSqOrig += o * o;
        localMaxErr = std::max(localMaxErr, err);
        localMinVal = std::min(localMinVal, o);
        localMaxVal = std::max(localMaxVal, o);
    }

    double globalSumSqErr, globalSumSqOrig, globalMaxErr, globalMinVal, globalMaxVal;
    size_t globalCount;
    size_t localCountLong = localSize;

    MPI_Allreduce(&localSumSqErr, &globalSumSqErr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localSumSqOrig, &globalSumSqOrig, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localMaxErr, &globalMaxErr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localMinVal, &globalMinVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMaxVal, &globalMaxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localCountLong, &globalCount, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

    double mse = globalSumSqErr / globalCount;
    double rmse = std::sqrt(mse);
    double l2norm = std::sqrt(globalSumSqOrig / globalCount);
    double nrmse = (l2norm > 0) ? (rmse / l2norm) : 0.0;
    double range = globalMaxVal - globalMinVal;
    double psnr = (mse > 0) ? (20 * std::log10(range) - 10 * std::log10(rmse)) : 999.0;

    if (rank == 0) {
        std::cout << "  Variable: " << varName << "\n";
        std::cout << "    NRMSE: " << std::scientific << std::setprecision(6) << nrmse << "\n";
        std::cout << "    PSNR : " << std::fixed << std::setprecision(2) << psnr << " dB\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0) {
            std::cout << "Usage: mpirun -n <np> ./decompress_mpi <original.bp> <compressed.bp> <output_decomp.bp> <decomp_dim> <var1> [var2] ...\n";
        }
        MPI_Finalize();
        return 0;
    }

    std::string origFile = argv[1];
    std::string compFile = argv[2];
    std::string decompOutFile = argv[3];
    size_t decompDim = std::stoul(argv[4]);

    std::vector<std::string> targetVars;
    for (int i = 5; i < argc; ++i) targetVars.push_back(argv[i]);

    if (rank == 0) {
        size_t origSize = getPathSize(origFile);
        size_t compSize = getPathSize(compFile);
        double ratio = (compSize > 0) ? (double)origSize / compSize : 0.0;
        std::cout << "========================================\n";
        std::cout << " Original:   " << origFile << "\n";
        std::cout << " Compressed: " << compFile << "\n";
        std::cout << " Output:     " << decompOutFile << "\n";
        std::cout << " Compression Ratio: " << std::fixed << std::setprecision(2) << ratio << "x\n";
        std::cout << "========================================\n";
    }

    try {
        adios2::ADIOS adios(MPI_COMM_WORLD);
        
        adios2::IO ioOrig = adios.DeclareIO("OrigReader");
        adios2::IO ioComp = adios.DeclareIO("CompReader");
        
        adios2::IO ioWrite = adios.DeclareIO("DecompWriter");
        ioWrite.SetEngine("BP5");

        adios2::Engine rOrig = ioOrig.Open(origFile, adios2::Mode::Read);
        adios2::Engine rComp = ioComp.Open(compFile, adios2::Mode::Read);
        
        adios2::Engine wDecomp = ioWrite.Open(decompOutFile, adios2::Mode::Write);

        int step = 0;
        while (rOrig.BeginStep() == adios2::StepStatus::OK) {
            if (rComp.BeginStep() != adios2::StepStatus::OK) break;
            
            wDecomp.BeginStep();

            if (rank == 0) std::cout << "\n[Step " << step << " Analysis]\n";

            auto vars = ioOrig.AvailableVariables();
            for (const auto& name : targetVars) {
                if (vars.find(name) == vars.end()) continue;
                
                std::string type = vars[name]["Type"];
                
                if (type == "double") analyze_and_write_step<double>(rOrig, rComp, wDecomp, ioOrig, ioComp, ioWrite, name, decompDim, rank, size);
                else if (type == "float") analyze_and_write_step<float>(rOrig, rComp, wDecomp, ioOrig, ioComp, ioWrite, name, decompDim, rank, size);
                // Will add different type handling
                //else if (type == "int32_t") analyze_and_write_step<int>(rOrig, rComp, wDecomp, ioOrig, ioComp, ioWrite, name, decompDim, rank, size);
            }

            rOrig.EndStep();
            rComp.EndStep();
            wDecomp.EndStep();
            
            step++;
        }
        rOrig.Close();
        rComp.Close();
        wDecomp.Close();

        if (rank == 0) std::cout << "\nDecompressed data saved to: " << decompOutFile << "\n";

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    MPI_Finalize();
    return 0;
}