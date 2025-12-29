#include <adios2.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

template<typename T>
void convert_variable_to_bin(
    adios2::Engine& reader,
    adios2::IO& readIO,
    const std::string& varName,
    const std::string& outFile
) {
    auto varRead = readIO.InquireVariable<T>(varName);
    if (!varRead) {
        std::cerr << "Error: Cannot inquire variable: " << varName << "\n";
        return;
    }

    auto shape = varRead.Shape();
    size_t ndims = shape.size();

    std::cout << "Original shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    size_t totalSize = 1;
    for (auto s : shape) totalSize *= s;

    std::vector<T> data(totalSize);
    reader.Get(varRead, data.data(), adios2::Mode::Sync);

    std::vector<size_t> newShape;
    size_t dimsToAdd = 5 - ndims;

    if (ndims >= 5) {
        newShape = shape;
    } else {
        for (size_t i = 0; i < dimsToAdd; ++i) {
            newShape.push_back(1);
        }
        for (auto s : shape) {
            newShape.push_back(s);
        }
    }

    std::cout << "Reshaped to: [";
    for (size_t i = 0; i < newShape.size(); ++i) {
        std::cout << newShape[i];
        if (i < newShape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::ofstream outStream(outFile, std::ios::binary);
    if (!outStream) {
        std::cerr << "Error: Cannot open output file: " << outFile << "\n";
        return;
    }

    outStream.write(reinterpret_cast<const char*>(data.data()), totalSize * sizeof(T));
    outStream.close();

    std::cout << "Successfully wrote " << totalSize << " elements to " << outFile << "\n";

    size_t fileSize = totalSize * sizeof(T);
    std::cout << "File size: " << (fileSize / (1024.0 * 1024.0)) << " MB\n";

    if (data.size() > 0) {
        T minVal = data[0];
        T maxVal = data[0];
        for (const auto& val : data) {
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
        std::cout << "Data range: [" << minVal << ", " << maxVal << "]\n";
    }
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout << "Usage: ./bin_mpi <input.bp> <variable_name> <output.bin>\n";
        std::cout << "\nExample:\n";
        std::cout << "  ./bin_mpi data.bp temperature output.bin.f32\n";
        std::cout << "\nDescription:\n";
        std::cout << "  Reads a variable from ADIOS2 BP file and writes it as flat binary.\n";
        std::cout << "  Automatically reshapes to 5D by prepending 1s:\n";
        std::cout << "    - 256x256x256x256 -> 1x256x256x256x256\n";
        std::cout << "    - 720x240x240    -> 1x1x720x240x240\n";
        return 0;
    }

    std::string inFile = argv[1];
    std::string varName = argv[2];
    std::string outFile = argv[3];

    std::cout << "=== ADIOS to Binary Converter ===\n";
    std::cout << "Input file:  " << inFile << "\n";
    std::cout << "Variable:    " << varName << "\n";
    std::cout << "Output file: " << outFile << "\n\n";

    try {
        adios2::ADIOS adios;
        adios2::IO readIO = adios.DeclareIO("Reader");

        adios2::Engine reader = readIO.Open(inFile, adios2::Mode::Read);

        bool variableFound = false;
        int step = 0;

       
        while (reader.BeginStep() == adios2::StepStatus::OK) {
            auto availableVars = readIO.AvailableVariables();

            if (step == 0) {
                std::cout << "Available variables:\n";
                for (const auto& [name, info] : availableVars) {
                    std::cout << "  - " << name << " (type: " << info.at("Type") << ")\n";
                }
                std::cout << "\n";
            }

            if (availableVars.find(varName) != availableVars.end()) {
                std::string type = availableVars[varName].at("Type");

                std::cout << "Processing variable: " << varName << " (type: " << type << ") from step " << step << "\n";

                if (type == "double") {
                    convert_variable_to_bin<double>(reader, readIO, varName, outFile);
                } else if (type == "float") {
                    convert_variable_to_bin<float>(reader, readIO, varName, outFile);
                } else if (type == "int32_t" || type == "int") {
                    convert_variable_to_bin<int32_t>(reader, readIO, varName, outFile);
                } else if (type == "int64_t") {
                    convert_variable_to_bin<int64_t>(reader, readIO, varName, outFile);
                } else {
                    std::cerr << "Error: Unsupported type '" << type << "' for variable " << varName << "\n";
                }

                variableFound = true;
                reader.EndStep();
                break;  
            }

            reader.EndStep();
            step++;
        }

        if (!variableFound) {
            std::cerr << "Error: Variable '" << varName << "' not found in any step of the file!\n";
            reader.Close();
            return 1;
        }

        reader.Close();

        std::cout << "\n=== Done! ===\n";

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
