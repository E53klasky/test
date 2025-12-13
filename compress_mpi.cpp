#include <adios2.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

template<typename T>
void compress_var(
    adios2::Engine& reader,
    adios2::Engine& writer,
    adios2::IO& readIO,
    adios2::IO& writeIO,
    const std::string& varName,
    size_t decompDim,
    int rank,
    int size,
    bool isFirstDefinition,
    adios2::Operator& op
) {
    auto varRead = readIO.InquireVariable<T>(varName);
    if (!varRead) {
        if (rank == 0) std::cerr << "    Cannot inquire variable: " << varName << "\n";
        return;
    }

    auto shape = varRead.Shape();
    size_t ndims = shape.size();

    if (decompDim >= ndims) {
        if (rank == 0) std::cerr << "    Error: Decomp dim " << decompDim << " too large for " << varName << " (ndims=" << ndims << ")\n";
        return;
    }

    std::vector<size_t> start(ndims, 0);
    std::vector<size_t> count = shape;

    count[decompDim] /= size;
    start[decompDim] = rank * count[decompDim];

    if (rank == size - 1) {
        count[decompDim] = shape[decompDim] - start[decompDim];
    }

    varRead.SetSelection({start, count});

    size_t localSize = 1;
    for (auto c : count) localSize *= c;
    std::vector<T> data(localSize);

    reader.Get(varRead, data.data(), adios2::Mode::Sync);

    // Define or inquire variable
    adios2::Variable<T> varWrite;
    if (isFirstDefinition) {
        varWrite = writeIO.DefineVariable<T>(varName, shape, start, count, adios2::ConstantDims);
        varWrite.AddOperation(op);
        
        if (rank == 0) {
            std::cout << "    [Define+Compress] " << varName << " shape=[";
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];
                if (i < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    } else {
        varWrite = writeIO.InquireVariable<T>(varName);
        if (rank == 0) std::cout << "    [Write] " << varName << "\n";
    }

    if (varWrite) {
        writer.Put(varWrite, data.data());
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0) {
            std::cout << "Usage: mpirun -n <np> ./compress_mpi <input.bp> <output.bp> <decomp_dim> <compressor> <error_bound> [var1] [var2] ...\n";
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
    for (int i = 6; i < argc; ++i) targetVars.push_back(argv[i]);

    try {
        adios2::ADIOS adios(MPI_COMM_WORLD);

        adios2::IO readIO = adios.DeclareIO("Reader");
        adios2::IO writeIO = adios.DeclareIO("Writer");
        writeIO.SetEngine("BP5");

        adios2::Params opParams;
        if (compressor == "CAESAR") {
            opParams = {{"error_bound", std::to_string(errorBound)}, {"batch_size", "32"}};
        } else if (compressor == "MGARD" || compressor == "ZFP" || compressor == "SZ") {
            opParams = {{"accuracy", std::to_string(errorBound)}};
        }
        auto op = adios.DefineOperator("Comp", compressor, opParams);

        adios2::Engine reader = readIO.Open(inFile, adios2::Mode::Read);
        adios2::Engine writer = writeIO.Open(outFile, adios2::Mode::Write);

        int step = 0;
        int totalVarsProcessed = 0;
        std::map<std::string, bool> varDefined;

        // Process each "step" (which may contain different variables)
        while (reader.BeginStep() == adios2::StepStatus::OK) {
            if (rank == 0) std::cout << "\n=== Step " << step << " ===\n";
            
            auto currentVars = readIO.AvailableVariables();
            
            if (rank == 0) {
                std::cout << "  Available variables in this step: ";
                for (const auto& v : currentVars) std::cout << v.first << " ";
                std::cout << "\n";
            }

            writer.BeginStep();

            // Process ALL variables in this step that match our target list
            for (const auto& [varName, varInfo] : currentVars) {
                // If user specified variables, only process those
                if (!targetVars.empty() && 
                    std::find(targetVars.begin(), targetVars.end(), varName) == targetVars.end()) {
                    continue;
                }

                std::string type = varInfo.at("Type");
                bool isFirst = (varDefined.find(varName) == varDefined.end());
                if (isFirst) varDefined[varName] = true;

                if (type == "double") {
                    compress_var<double>(reader, writer, readIO, writeIO, varName, 
                                        decompDim, rank, size, isFirst, op);
                    if (isFirst) totalVarsProcessed++;
                } else if (type == "float") {
                    compress_var<float>(reader, writer, readIO, writeIO, varName, 
                                       decompDim, rank, size, isFirst, op);
                    if (isFirst) totalVarsProcessed++;
                } else {
                    if (rank == 0) {
                        std::cout << "    Skipping " << varName << " (unsupported type: " << type << ")\n";
                    }
                }
            }

            writer.EndStep();
            reader.EndStep();
            step++;
        }

        reader.Close();
        writer.Close();

        if (rank == 0) {
            std::cout << "\n=== Done! ===\n";
            std::cout << "Processed " << step << " steps\n";
            std::cout << "Compressed " << totalVarsProcessed << " variables total\n";
            std::cout << "Output: " << outFile << "\n";
        }

    } catch (std::exception& e) {
        std::cerr << "[Rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}
