#include <adios2.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sys/stat.h>

template<typename T>
struct MetricsResult {
    double compressionRatio;
    double nrmse;
    double linf;
    size_t originalBytes;
    size_t compressedBytes;
    size_t numElements;
};

template<typename T>
MetricsResult<T> computeMetrics(
    const std::vector<T>& original ,
    const std::vector<T>& compressed ,
    size_t originalBytes ,
    size_t compressedBytes
) {
    MetricsResult<T> result;
    result.originalBytes = originalBytes;
    result.compressedBytes = compressedBytes;
    result.numElements = original.size();

    if (original.size() != compressed.size()) {
        result.compressionRatio = 0.0;
        result.nrmse = -1.0;
        result.linf = -1.0;
        return result;
    }

    result.compressionRatio = static_cast<double>(originalBytes) / static_cast<double>(compressedBytes);

    double sumSquaredError = 0.0;
    double sumSquaredOriginal = 0.0;
    double maxAbsError = 0.0;

    for (size_t i = 0; i < original.size(); ++i) {
        double error = std::abs(static_cast<double>(original[i]) - static_cast<double>(compressed[i]));
        double origVal = static_cast<double>(original[i]);

        sumSquaredError += error * error;
        sumSquaredOriginal += origVal * origVal;
        maxAbsError = std::max(maxAbsError , error);
    }

    result.nrmse = sumSquaredError;
    result.linf = maxAbsError;

    return result;
}

size_t getFileSize(const std::string& filename) {
    struct stat stat_buf;
    int rc = stat(filename.c_str() , &stat_buf);
    return rc == 0 ? stat_buf.st_size : 0;
}

template<typename T>
bool analyzeVariable(
    adios2::ADIOS& adios ,
    const std::string& uncompressedFile ,
    const std::string& compressedFile ,
    const std::string& varName ,
    size_t decompDim ,
    int rank ,
    int size ,
    int maxSteps = -1
) {
    adios2::IO readIOUncomp = adios.DeclareIO("ReadUncomp_" + varName + "_" + std::to_string(rank));
    adios2::Engine readerUncomp = readIOUncomp.Open(uncompressedFile , adios2::Mode::Read);

    adios2::IO readIOComp = adios.DeclareIO("ReadComp_" + varName + "_" + std::to_string(rank));
    adios2::Engine readerComp = readIOComp.Open(compressedFile , adios2::Mode::Read);

    if (rank == 0)
        std::cout << "\n=== Analyzing " << varName << " ===\n";

    int step = 0;
    double totalCompressionRatio = 0.0;
    double totalNRMSE = 0.0;
    double totalLinf = 0.0;
    int successfulSteps = 0;

    while (readerUncomp.BeginStep() == adios2::StepStatus::OK &&
        readerComp.BeginStep() == adios2::StepStatus::OK &&
        (maxSteps < 0 || step < maxSteps))
    {
        adios2::Variable<T> varUncomp = readIOUncomp.InquireVariable<T>(varName);
        adios2::Variable<T> varComp = readIOComp.InquireVariable<T>(varName);

        if (!varUncomp || !varComp) {
            if (rank == 0)
                std::cerr << "ERROR: Could not find variable in one of the files\n";
            readerUncomp.EndStep();
            readerComp.EndStep();
            continue;
        }

        auto shape = varUncomp.Shape();
        size_t ndims = shape.size();

        if (rank == 0) {
            std::cout << "Step " << step << ": Shape [";
            for (size_t i = 0; i < shape.size(); i++) {
                std::cout << shape[i];
                if (i < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }

        if (decompDim >= ndims) {
            if (rank == 0)
                std::cerr << "ERROR: Decomp dim " << decompDim
                << " exceeds variable dims " << ndims << "\n";
            readerUncomp.EndStep();
            readerComp.EndStep();
            step++;
            continue;
        }

        std::vector<size_t> start(ndims , 0);
        std::vector<size_t> count = shape;

        count[decompDim] /= size;
        start[decompDim] = rank * count[decompDim];

        varUncomp.SetSelection({ start, count });
        varComp.SetSelection({ start, count });

        size_t localSize = 1;
        for (auto c : count) localSize *= c;

        std::vector<T> dataUncomp(localSize);
        std::vector<T> dataComp(localSize);

        readerUncomp.Get(varUncomp , dataUncomp.data());
        readerComp.Get(varComp , dataComp.data());

        readerUncomp.EndStep();
        readerComp.EndStep();


        size_t originalBytes = localSize * sizeof(T);
        size_t compressedBytes = localSize * sizeof(T) / 2;

        auto localMetrics = computeMetrics(dataUncomp , dataComp , originalBytes , compressedBytes);

        double globalSumSquaredError = 0.0;
        double globalSumSquaredOriginal = 0.0;
        double globalMaxError = 0.0;
        size_t globalElements = 0;

        MPI_Allreduce(&localMetrics.nrmse , &globalSumSquaredError , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
        MPI_Allreduce(&localMetrics.linf , &globalMaxError , 1 , MPI_DOUBLE , MPI_MAX , MPI_COMM_WORLD);
        MPI_Allreduce(&localSize , &globalElements , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , MPI_COMM_WORLD);

        double localSumSquaredOriginal = 0.0;
        for (size_t i = 0; i < dataUncomp.size(); ++i) {
            double val = static_cast<double>(dataUncomp[i]);
            localSumSquaredOriginal += val * val;
        }
        MPI_Allreduce(&localSumSquaredOriginal , &globalSumSquaredOriginal , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);

        double rmse = std::sqrt(globalSumSquaredError / globalElements);
        double norm = std::sqrt(globalSumSquaredOriginal / globalElements);
        double globalNRMSE = (norm > 0.0) ? (rmse / norm) : 0.0;

        size_t totalOrigBytes = globalElements * sizeof(T);
        size_t totalCompBytes = totalOrigBytes / 2;
        double globalCompressionRatio = static_cast<double>(totalOrigBytes) / static_cast<double>(totalCompBytes);

        if (rank == 0) {
            std::cout << "  Step " << step << ":\n";
            std::cout << "    Total Elements: " << globalElements << "\n";
            std::cout << "    Compression Ratio: " << std::fixed << std::setprecision(2)
                << globalCompressionRatio << "x\n";
            std::cout << "    NRMSE: " << std::scientific << std::setprecision(6)
                << globalNRMSE << "\n";
            std::cout << "    L∞ Error: " << std::scientific << std::setprecision(6)
                << globalMaxError << "\n";
        }

        totalCompressionRatio += globalCompressionRatio;
        totalNRMSE += globalNRMSE;
        totalLinf = std::max(totalLinf , globalMaxError);
        successfulSteps++;
        step++;
    }

    readerUncomp.Close();
    readerComp.Close();

    if (rank == 0 && successfulSteps > 0) {
        std::cout << "\n  Summary for " << varName << " (" << successfulSteps << " steps):\n";
        std::cout << "    Avg Compression Ratio: " << std::fixed << std::setprecision(2)
            << (totalCompressionRatio / successfulSteps) << "x\n";
        std::cout << "    Avg NRMSE: " << std::scientific << std::setprecision(6)
            << (totalNRMSE / successfulSteps) << "\n";
        std::cout << "    Max L∞ Error: " << std::scientific << std::setprecision(6)
            << totalLinf << "\n";
    }

    return successfulSteps > 0;
}

int main(int argc , char** argv)
{
    MPI_Init(&argc , &argv);

    int rank , size;
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    MPI_Comm_size(MPI_COMM_WORLD , &size);

    if (argc < 5) {
        if (rank == 0) {
            std::cout << "Usage:\n"
                << "  mpirun -n <np> ./prog <uncompressed.bp> <compressed.bp> <decomp_dim> <var1> [var2] ...\n\n"
                << "Example:\n"
                << "  mpirun -n 4 ./prog original.bp compressed.bp 0 wave_3d temperature\n\n"
                << "Note: Variables should exist in both files with same names\n";
        }
        MPI_Finalize();
        return 0;
    }

    std::string uncompressedFile = argv[1];
    std::string compressedFile = argv[2];
    size_t decompDim = std::stoul(argv[3]);

    std::vector<std::string> targetVars;
    for (int i = 4; i < argc; ++i) {
        targetVars.push_back(argv[i]);
    }

    if (targetVars.empty()) {
        if (rank == 0)
            std::cerr << "ERROR: Please specify at least one variable to analyze\n";
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::time_t now = std::time(nullptr);
        std::tm* localTime = std::localtime(&now);

        std::cout << "========================================\n";
        std::cout << "Date/Time   : " << std::put_time(localTime , "%Y-%m-%d %H:%M:%S") << "\n";
        std::cout << "Uncompressed: " << uncompressedFile << "\n";
        std::cout << "Compressed  : " << compressedFile << "\n";
        std::cout << "Decomp Dim  : " << decompDim << "\n";
        std::cout << "MPI Ranks   : " << size << "\n";
        std::cout << "Variables   : ";
        for (size_t i = 0; i < targetVars.size(); ++i) {
            std::cout << targetVars[i];
            if (i < targetVars.size() - 1) std::cout << ", ";
        }
        std::cout << "\n========================================\n";

        size_t uncompSize = getFileSize(uncompressedFile);
        size_t compSize = getFileSize(compressedFile);

        if (uncompSize > 0 && compSize > 0) {
            double overallRatio = static_cast<double>(uncompSize) / static_cast<double>(compSize);
            std::cout << "\nOverall File Sizes:\n";
            std::cout << "  Uncompressed: " << (uncompSize / 1024.0 / 1024.0) << " MB\n";
            std::cout << "  Compressed  : " << (compSize / 1024.0 / 1024.0) << " MB\n";
            std::cout << "  File Compression Ratio: " << std::fixed << std::setprecision(2)
                << overallRatio << "x\n";
        }
    }

    try {
        adios2::ADIOS adios(MPI_COMM_WORLD);

        adios2::IO probeIO = adios.DeclareIO("ProbeIO");
        adios2::Engine probe = probeIO.Open(uncompressedFile , adios2::Mode::Read);
        probe.BeginStep();
        auto allVars = probeIO.AvailableVariables();
        probe.EndStep();
        probe.Close();

        if (allVars.empty()) {
            if (rank == 0) std::cout << "No variables found.\n";
            MPI_Finalize();
            return 0;
        }

        for (const auto& varName : targetVars) {
            if (allVars.find(varName) == allVars.end()) {
                if (rank == 0)
                    std::cerr << "\nERROR: Variable '" << varName << "' not found in uncompressed file\n";
                continue;
            }

            std::string type = allVars[varName]["Type"];

            if (rank == 0)
                std::cout << "\nDetected type for " << varName << ": " << type << "\n";

            MPI_Barrier(MPI_COMM_WORLD);

            bool success = false;
            if (type == "double")
                success = analyzeVariable<double>(adios , uncompressedFile , compressedFile , varName , decompDim , rank , size);
            else if (type == "float")
                success = analyzeVariable<float>(adios , uncompressedFile , compressedFile , varName , decompDim , rank , size);
            else if (type == "int32_t" || type == "int")
                success = analyzeVariable<int>(adios , uncompressedFile , compressedFile , varName , decompDim , rank , size);
            else if (type == "uint32_t" || type == "unsigned int")
                success = analyzeVariable<unsigned int>(adios , uncompressedFile , compressedFile , varName , decompDim , rank , size);
            else if (type == "int64_t" || type == "long long")
                success = analyzeVariable<long long>(adios , uncompressedFile , compressedFile , varName , decompDim , rank , size);
            else if (type == "uint64_t" || type == "unsigned long long")
                success = analyzeVariable<unsigned long long>(adios , uncompressedFile , compressedFile , varName , decompDim , rank , size);
            else {
                if (rank == 0)
                    std::cerr << "ERROR: Unsupported type: " << type << "\n";
            }

            if (!success && rank == 0) {
                std::cerr << "Failed to analyze variable: " << varName << "\n";
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0) {
            std::cout << "\n========================================\n";
            std::cout << "Analysis complete!\n";
            std::cout << "========================================\n";
        }

    }
    catch (std::exception& e) {
        std::cerr << "[Rank " << rank << "] ERROR: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}