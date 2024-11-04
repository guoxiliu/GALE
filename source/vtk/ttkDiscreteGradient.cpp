#include <Gale.h>
#include <MemoryUsage.h>
#include <DiscreteGradient.h>
#include <vtkXMLGenericDataObjectReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <iostream>
#include <unistd.h>

#include "ttkUtils.h"


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " -i <input.vtu>" << std::endl;
        std::cout << "Missing required input .vtu file!" << std::endl;
        return EXIT_FAILURE;
    }

    int c;
    int blockNumber = 0;
    int blockSize = 0;
    int threadNumber = 4;
    int numThreadsPerCluster = 1;
    std::string fileName = "";
    std::string inputArrayName = "";
    vtkDataArray *inputScalarField = nullptr;

    while ((c = getopt(argc, argv, "a:b:i:p:s:t:")) != -1) {
        switch (c) {
            case 'i':
                fileName = optarg;
                break;
            case 'a':
                inputArrayName = optarg;
                break;
            case 'b':
                blockNumber = atoi(optarg);
                break;
            case 'p':
                numThreadsPerCluster = atoi(optarg);
                break;
            case 's':
                blockSize = atoi(optarg);
                break;
            case 't':
                threadNumber = atoi(optarg);
                break;
        }
    }

    std::cout << "Parsing vtu file: " << fileName << std::endl;
    
    //read all the data from the file
    vtkSmartPointer<vtkXMLGenericDataObjectReader> reader =
        vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
    reader->SetFileName(fileName.c_str());
    reader->Update();
    
    int scalarFieldIndex = 0;
    vtkDataObject *inputDataObject = reader->GetOutput();
    if(!inputDataObject) {
        std::cerr << "Unable to read input file `" << fileName << "' :(" << std::endl;
        return 1;
    }
    vtkDataSet *inputDataSet = vtkDataSet::SafeDownCast(inputDataObject);
    vtkPointData *pointData = inputDataSet->GetPointData();
    // get the scalar field array to process 
    if (!inputArrayName.empty()) {
        inputScalarField = pointData->GetArray(inputArrayName.c_str());
        if (inputScalarField == nullptr) {
            std::cout << "Cannot find the array '" << inputArrayName << "' in point data" << std::endl;
            inputScalarField = inputDataSet->GetCellData()->GetArray(inputArrayName.c_str());
            if (inputScalarField == nullptr) {
                std::cout << "Cannot find the array '" << inputArrayName << "' in cell data" << std::endl;
                return EXIT_FAILURE;
            }
            
        }
        for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
            if (pointData->GetArray(i)->GetName() == inputArrayName) {
                scalarFieldIndex = i;
                break;
            }
        }
        std::cout << "Processing array: '" << inputArrayName << "'" << std::endl;
    }
    // default to the first scalar field array 
    else {
        inputScalarField = pointData->GetArray(scalarFieldIndex);
        std::cout << "Default to array: '" << inputScalarField->GetName() << "'" << std::endl;
    }

    MemoryUsage m;

    ttk::Gale gale(threadNumber, numThreadsPerCluster);
    if (blockNumber > 0 && blockSize > 0)
        gale.initialize(inputDataSet, blockNumber, blockSize);
    else if (blockNumber > 0)
        gale.initialize(inputDataSet, blockNumber);
    else 
        gale.initialize(inputDataSet);

    ttk::dcg::DiscreteGradient dgPlugin{};
    dgPlugin.setDebugLevel(3);
    dgPlugin.setThreadNumber(threadNumber);
    dgPlugin.preconditionTriangulationSimplified(&gale);
    
    vtkDataArray *offsetField = GetOrderArray(inputDataSet, scalarFieldIndex, 1);
    dgPlugin.setInputScalarField(GetVoidPointer(inputScalarField));
    dgPlugin.setInputOffsets(static_cast<SimplexId *>(GetVoidPointer(offsetField)));

    int res = dgPlugin.buildGradient(gale);
    if (res != 0) {
        std::cerr << "Execution failed, status code: " << res << std::endl;
    }

    std::cout << "[DiscreteGradient] Memory usage: " << m.getValue_in_MB(false) << " MB." << std::endl;

    return 0;
}
