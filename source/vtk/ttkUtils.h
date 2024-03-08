#pragma once

#include <OrderDisambiguation.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkType.h>

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

