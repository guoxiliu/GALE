#include <ExplicitTriangulation.h>
#include <TestTopoRelations.h>
#include <ScalarFieldCriticalPoints.h>
#include <OrderDisambiguation.h>
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <iostream>
#include <stdio.h>
#include <unistd.h>

/// Retrieve pointer to the internal data
/// This method is a workaround to emulate
/// the old GetVoidPointer in vtkDataArray
void *GetVoidPointer(vtkDataArray *array, vtkIdType start = 0) {
    void *outPtr = nullptr;
    if(array == nullptr)
    return outPtr;

    switch(array->GetDataType()) {
    vtkTemplateMacro(
        auto *aosArray = vtkAOSDataArrayTemplate<VTK_TT>::FastDownCast(array);
        if(aosArray) { outPtr = aosArray->GetVoidPointer(start); });
    }
    return outPtr;
}


std::string GetOrderArrayName(vtkDataArray *const array) {
    return std::string(array->GetName()) + "_Order";
}

vtkDataArray *GetOrderArray(vtkDataSet *const inputData,
                                        const int scalarArrayIdx,
                                        const int orderArrayIdx = 0) {

    auto isValidOrderArray = [](vtkDataArray *const array) {
        if(!array)
            return -4;

        if(array->GetNumberOfComponents() != 1)
            return -3;

        auto temp = vtkSmartPointer<vtkIntArray>::New();
        if(array->GetDataType() != temp->GetDataType())
            return -2;

        const std::string name(array->GetName());
        if(name.size() < 6 || (name.rfind("_Order") != (name.size() - 6)))
            return -1;

        return 1;
    };

    auto scalarArray = inputData->GetPointData()->GetArray(scalarArrayIdx);
    if(!scalarArray) {
        std::cerr << "Unable to retrieve input scalar array for idx " << scalarArrayIdx << std::endl;
        return nullptr;
    } else if(isValidOrderArray(scalarArray) == 1) {
        std::cout << "Retrieved scalar array `" << scalarArray->GetName() << "` is already an order array." << std::endl;
        return scalarArray;
    }

    auto orderArray = inputData->GetAttributesAsFieldData(scalarArrayIdx)
                                ->GetArray(GetOrderArrayName(scalarArray).data());

    switch(isValidOrderArray(orderArray)) {
        case -4: {
            std::cout << "[WARNING] No pre-existing order for array:" << std::endl;
            std::cout << "[WARNING]   `" << std::string(scalarArray->GetName()) << "`." << std::endl;

            std::cout << "Initializing order array." << std::endl;

            auto nVertices = scalarArray->GetNumberOfTuples();
            auto newOrderArray = vtkSmartPointer<vtkIntArray>::New();
            newOrderArray->SetName(GetOrderArrayName(scalarArray).data());
            newOrderArray->SetNumberOfComponents(1);
            newOrderArray->SetNumberOfTuples(nVertices);

            switch(scalarArray->GetDataType()) {
                vtkTemplateMacro(ttk::preconditionOrderArray(nVertices,
                    static_cast<VTK_TT *>(GetVoidPointer(scalarArray)),
                    static_cast<ttk::SimplexId *>(GetVoidPointer(newOrderArray)), 10));
            }

            // append order array temporarily to input
            inputData->GetAttributesAsFieldData(scalarArrayIdx)->AddArray(newOrderArray);

            std::cout << "TIP: run `ttkArrayPreconditioning` first" << std::endl;
            std::cout << "for improved performances :)" << std::endl;

            return newOrderArray;
        }

        case -3: {
            std::cerr << 
            "Retrieved order array `" << std::string(orderArray->GetName())
            << ">>` for scalar array `" << std::string(scalarArray->GetName())
            << ">>` has more than one component." << std::endl;
            return nullptr;
        }

        case -2: {
            std::cerr << 
            "Retrieved order array `" << std::string(orderArray->GetName())
            << ">>` for scalar array `" << std::string(scalarArray->GetName())
            << ">>` is of incorrect type." << std::endl;
            auto temp = vtkSmartPointer<vtkIntArray>::New();
            std::cerr << " -> use `ttkArrayEditor` to convert data type to `"
                            << std::string(temp->GetDataTypeAsString()) << ">>`." << std::endl;
            return nullptr;
        }

        default: {
            std::cout << "Retrieved order array `" << std::string(orderArray->GetName())
                    << ">>` for scalar array `" << std::string(scalarArray->GetName()) << ">>`." << std::endl;
            return orderArray;
        }
    }
}


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
