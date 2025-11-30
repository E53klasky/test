#include <adios2.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>

template<typename T>
void processVariable(
    adios2::ADIOS& adios ,
    const std::string& inFile ,
    const std::string& outFile ,
    const std::string& varName ,
    size_t decompDim ,
    adios2::Operator& op ,
    int rank ,
    int size
) {
    adios2::IO readIO = adios.DeclareIO("ReadIO_" + varName + "_" + std::to_string(rank));
    adios2::Engine reader = readIO.Open(inFile , adios2::Mode::Read);

    adios2::IO writeIO = adios.DeclareIO("WriteIO_" + varName + "_" + std::to_string(rank));
    writeIO.SetEngine("BP5");
    adios2::Engine writer = writeIO.Open(outFile , adios2::Mode::Write);

    if (rank == 0)
        std::cout << "\n=== Processing " << varName << " variable ===\n";

    int step = 0;
    while (reader.BeginStep() == adios2::StepStatus::OK)
    {
        adios2::Variable<T> varRead = readIO.InquireVariable<T>(varName);

        if (!varRead) {
            if (rank == 0)
                std::cerr << "ERROR: Could not find " << varName << " variable\n";
            reader.EndStep();
            continue;
        }

        auto shape = varRead.Shape();
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
            reader.EndStep();
            step++;
            continue;
        }

        std::vector<size_t> start(ndims , 0);
        std::vector<size_t> count = shape;

        count[decompDim] /= size;
        start[decompDim] = rank * count[decompDim];

        varRead.SetSelection({ start, count });

        size_t localSize = 1;
        for (auto c : count) localSize *= c;

        std::vector<T> data(localSize);

        reader.Get(varRead , data.data());
        reader.EndStep();

        if (rank == 0) {
            std::cout << "  Each rank read " << localSize << " elements (local chunk)\n";
        }

        if (step == 0) {
            auto varWrite = writeIO.DefineVariable<T>(
                varName ,
                shape ,
                start ,
                count ,
                adios2::ConstantDims
            );

            varWrite.AddOperation(op);
            if (rank == 0) {
                std::cout << "   Added compression to " << varName << "\n";
                std::cout << "   Global shape: [";
                for (size_t i = 0; i < shape.size(); i++) {
                    std::cout << shape[i];
                    if (i < shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]\n";
                std::cout << "   Rank 0 writes: start=[";
                for (size_t i = 0; i < start.size(); i++) {
                    std::cout << start[i];
                    if (i < start.size() - 1) std::cout << ", ";
                }
                std::cout << "], count=[";
                for (size_t i = 0; i < count.size(); i++) {
                    std::cout << count[i];
                    if (i < count.size() - 1) std::cout << ", ";
                }
                std::cout << "]\n";
            }
        }

        auto varWrite = writeIO.InquireVariable<T>(varName);
        writer.BeginStep();
        writer.Put(varWrite , data.data());
        writer.EndStep();

        if (rank == 0)
            std::cout << "  Compressed and wrote step " << step << "\n";
        step++;
    }

    if (rank == 0)
        std::cout << "\nSuccessfully compressed " << step << " timesteps of " << varName << "\n";

    reader.Close();
    writer.Close();
}

int main(int argc , char** argv)
{
    MPI_Init(&argc , &argv);

    int rank , size;
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    MPI_Comm_size(MPI_COMM_WORLD , &size);

    if (argc < 6)
    {
        if (rank == 0)
        {
            std::cout << "Usage:\n"
                << "  mpirun -n <np> ./prog <input.bp> <output.bp> <decomp_dim> <compressor> <error_bound> [var1] [var2] ...\n\n"
                << "Example:\n"
                << "  mpirun -n 4 ./prog in.bp out.bp 0 CAESAR 0.001 wave_3d\n"
                << "  mpirun -n 4 ./prog in.bp out.bp 1 ZFP 0.01 temperature pressure\n\n"
                << "Compressors: CAESAR, MGARD, ZFP, SZ3\n"
                << "Note: If no variables specified, compresses ALL variables\n";
        }
        MPI_Finalize();
        return 0;
    }

    std::string inFile = argv[1];
    std::string outFile = argv[2];
    size_t decompDim = std::stoul(argv[3]);
    std::string compressor = argv[4];
    float errorBound = std::stof(argv[5]);

    std::vector<std::string> targetVars;
    for (int i = 6; i < argc; ++i) {
        targetVars.push_back(argv[i]);
    }

    if (rank == 0)
    {
        std::cout << "=== Compression Settings ===\n";
        std::cout << "Input        : " << inFile << "\n";
        std::cout << "Output       : " << outFile << "\n";
        std::cout << "Compressor   : " << compressor << "\n";
        std::cout << "Error Bound  : " << errorBound << "\n";
        std::cout << "Decomp Dim   : " << decompDim << "\n";
        std::cout << "MPI Ranks    : " << size << "\n";
        if (targetVars.empty()) {
            std::cout << "Variables    : ALL\n";
        }
        else {
            std::cout << "Variables    : ";
            for (size_t i = 0; i < targetVars.size(); ++i) {
                std::cout << targetVars[i];
                if (i < targetVars.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
        }
        std::cout << "=============================\n";
    }

    try {
        adios2::ADIOS adios(MPI_COMM_WORLD);

        adios2::Params opParams;
        if (compressor == "CAESAR")
        {
            opParams = {
                {"error_bound", std::to_string(errorBound)},
                {"mode", "CAESAR_V"},
                {"batch_size", "32"}
            };
        }
        else if (compressor == "MGARD")
        {
            opParams = { {"accuracy", std::to_string(errorBound)} };
        }
        else if (compressor == "ZFP")
        {
            opParams = { {"accuracy", std::to_string(errorBound)} };
        }
        else if (compressor == "SZ")
        {
            opParams = {
                {"accuracy", std::to_string(errorBound)}
            };
        }
        else
        {
            if (rank == 0)
                std::cerr << "Unknown compressor: " << compressor << "\n";
            MPI_Finalize();
            return 0;
        }

        auto op = adios.DefineOperator("Comp" , compressor , opParams);

        adios2::IO probeIO = adios.DeclareIO("ProbeIO");
        adios2::Engine probe = probeIO.Open(inFile , adios2::Mode::Read);
        probe.BeginStep();
        auto allVars = probeIO.AvailableVariables();
        probe.EndStep();
        probe.Close();

        if (allVars.empty())
        {
            if (rank == 0) std::cout << "No variables to compress.\n";
            MPI_Finalize();
            return 0;
        }

        std::vector<std::string> varsToProcess;
        if (targetVars.empty()) {
            for (const auto& varPair : allVars) {
                varsToProcess.push_back(varPair.first);
            }
        }
        else {
            varsToProcess = targetVars;
        }

        for (const auto& varName : varsToProcess)
        {
            if (allVars.find(varName) == allVars.end()) {
                if (rank == 0)
                    std::cerr << "\nERROR: Variable '" << varName << "' not found\n";
                continue;
            }

            std::string type = allVars[varName]["Type"];

            if (rank == 0)
                std::cout << "\nDetected type for " << varName << ": " << type << "\n";

            MPI_Barrier(MPI_COMM_WORLD);

            if (type == "double")
                processVariable<double>(adios , inFile , outFile , varName , decompDim , op , rank , size);
            else if (type == "float")
                processVariable<float>(adios , inFile , outFile , varName , decompDim , op , rank , size);
            else if (type == "int32_t" || type == "int")
                processVariable<int>(adios , inFile , outFile , varName , decompDim , op , rank , size);
            else if (type == "uint32_t" || type == "unsigned int")
                processVariable<unsigned int>(adios , inFile , outFile , varName , decompDim , op , rank , size);
            else if (type == "int64_t" || type == "long long")
                processVariable<long long>(adios , inFile , outFile , varName , decompDim , op , rank , size);
            else if (type == "uint64_t" || type == "unsigned long long")
                processVariable<unsigned long long>(adios , inFile , outFile , varName , decompDim , op , rank , size);
            else
            {
                if (rank == 0)
                    std::cerr << "Unsupported type: " << type << ". Skipping.\n";
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == 0)
        {
            std::cout << "\n========================================\n";
            std::cout << "Compression complete!\n";
            std::cout << "Output: " << outFile << "\n";
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