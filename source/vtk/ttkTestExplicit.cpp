#include <ExplicitTriangulation.h>
#include <TestTopoRelations.h>
#include <ScalarFieldCriticalPoints.h>
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
    int threadNumber = 4;
    std::string fileName = "";
    std::string inputArrayName = "";
    vtkDataArray *inputScalarField = nullptr;

    while ((c = getopt(argc, argv, "a:i:t:")) != -1) {
        switch (c) {
            case 'i':
                fileName = optarg;
                break;
            case 'a':
                inputArrayName = optarg;
                break;
            case 't':
                threadNumber = atoi(optarg);
                break;
        }
    }

    std::cout << "Parsing vtu file: " << fileName << std::endl;
    
    //read all the data from the file
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
        vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(fileName.c_str());
    reader->Update();

    vtkUnstructuredGrid *unstructuredGrid = reader->GetOutput();
    ttk::ExplicitTriangulation explicitTriangulation;
    explicitTriangulation.createExplicitTriangulation(unstructuredGrid);

    // // topo relations plugin
    // ttk::TestTopoRelations testPlugin;
    // testPlugin.preconditionTriangulation(&explicitTriangulation);
    // testPlugin.setThreadNumber(threadNumber);
    // testPlugin.execute(&explicitTriangulation);

    std::vector<std::pair<ttk::SimplexId, char>> criticalPoints_;

    ttk::ScalarFieldCriticalPoints cpPlugin;
    cpPlugin.preconditionTriangulation(&explicitTriangulation);
    cpPlugin.setThreadNumber(threadNumber);
    cpPlugin.setOutput(&criticalPoints_);

    int scalarFieldIndex = 0;
    vtkPointData *pointData = unstructuredGrid->GetPointData();
    // get the scalar field array to process 
    if (!inputArrayName.empty()) {
        inputScalarField = pointData->GetArray(inputArrayName.c_str());
        if (inputScalarField == nullptr) {
            std::cout << "Cannot find the array '" << inputArrayName << "' in point data" << std::endl;
            inputScalarField = unstructuredGrid->GetCellData()->GetArray(inputArrayName.c_str());
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
    
    vtkDataArray *offsetField = GetOrderArray(reader->GetOutput(), scalarFieldIndex, 1);

    int res = cpPlugin.executeLegacy(static_cast<ttk::SimplexId *>(GetVoidPointer(offsetField)), &explicitTriangulation);
    if (res != 0) {
        std::cerr << "Execution failed, status code: " << res << std::endl;
    }

    return 0;
}
