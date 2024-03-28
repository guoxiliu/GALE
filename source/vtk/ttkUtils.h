#pragma once

#include <OrderDisambiguation.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkType.h>

class vtkIdTypeArray;
class vtkIntArray;
class vtkDataArray;
class vtkCellArray;

#ifdef TTK_ENABLE_64BIT_IDS
using ttkSimplexIdTypeArray = vtkIdTypeArray;
#else
using ttkSimplexIdTypeArray = vtkIntArray;
#endif

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

void SetVoidArray(vtkDataArray *array,
                    void *data,
                    vtkIdType size,
                    int save) {
  switch(array->GetDataType()) {
    vtkTemplateMacro(
      auto *aosArray = vtkAOSDataArrayTemplate<VTK_TT>::FastDownCast(array);
      if(aosArray) { aosArray->SetVoidArray(data, size, save); } else {
        std::cerr << "SetVoidArray on incompatible vtkDataArray:" << endl;
        array->Print(std::cerr);
      });
  }
}

std::string GetOrderArrayName(vtkDataArray *const array) {
    return std::string(array->GetName()) + "_Order";
}

vtkDataArray *GetOrderArray(vtkDataSet *const inputData,
                            const int scalarArrayIdx,
                            const int orderArrayIdx,
                            const bool enforceOrderArrayIdx = false) {

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

    if(enforceOrderArrayIdx) {
        auto orderArray = inputData->GetPointData()->GetArray(orderArrayIdx);
        switch(isValidOrderArray(orderArray)) {
            case -4: {
                std::cerr << "Unable to retrieve enforced order array at idx " << orderArrayIdx << std::endl;
                return nullptr;
            }
            case -3: {
                std::cerr << "Retrieved enforced order array `" << orderArray->GetName() << 
                             "` has more than one component." << std::endl;
                return nullptr;
            }
            case -2: {
                std::cerr << "Enforced order array `" << orderArray->GetName() << 
                             "` is of incorrect type." << std::endl;
                auto temp = vtkSmartPointer<ttkSimplexIdTypeArray>::New();
                std::cerr << " -> use `ttkArrayEditor` to convert data type to `" << 
                             temp->GetDataTypeAsString() << "`." << std::endl;
                return nullptr;
            }
            default: {
                std::cout << "Retrieved enforced order array `" << orderArray->GetName() << "`." << std::endl;
                return orderArray;
            }
        }
    }
    auto scalarArray = inputData->GetPointData()->GetArray(scalarArrayIdx);
    if(!scalarArray) {
        std::cerr << "Unable to retrieve input scalar array for idx " << scalarArrayIdx << std::endl;
        return nullptr;
    } else if(isValidOrderArray(scalarArray) == 1) {
        std::cout << "Retrieved scalar array `" << scalarArray->GetName() << "` is already an order array." << std::endl;
        return scalarArray;
    }

    auto orderArray = inputData->GetAttributesAsFieldData(vtkDataObject::AttributeTypes::POINT)
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
            inputData->GetAttributesAsFieldData(vtkDataObject::AttributeTypes::POINT)->AddArray(newOrderArray);

            std::cout << "TIP: run `ttkArrayPreconditioning` first for improved performances :)" << std::endl;

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

vtkDataArray *GetInputArray(std::string arrayName, vtkDataSet *dataSet, int &arrayIndex) {
    vtkPointData *pointData = dataSet->GetPointData();
    vtkDataArray *inputArray = nullptr;
    // get the scalar field array to process 
    if (!arrayName.empty()) {
        inputArray = pointData->GetArray(arrayName.c_str());
        if (inputArray == nullptr) {
            std::cout << "Cannot find the array '" << arrayName << "' in point data" << std::endl;
            inputArray = dataSet->GetCellData()->GetArray(arrayName.c_str());
            if (inputArray == nullptr) {
                std::cout << "Cannot find the array '" << arrayName << "' in cell data" << std::endl;
                return inputArray;
            }
        }
        for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
            if (pointData->GetArray(i)->GetName() == arrayName) {
                arrayIndex = i;
                break;
            }
        }
        std::cout << "[GetInputArray] Processing array: '" << arrayName << "', array index: " << arrayIndex << std::endl;
    }
    // default to the first scalar field array 
    else {
        inputArray = pointData->GetArray(arrayIndex);
        std::cout << "Default to array: '" << inputArray->GetName() << "'" << std::endl;
    }
    return inputArray;
}
