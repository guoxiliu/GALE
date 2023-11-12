#include "ttk/AcTopo.h"
#include "ttk/ExplicitTriangulation.h"
#include "ttk/TestTopoRelations.h"
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <iostream>

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " -i <input.vtu>" << std::endl;
        std::cout << "Missing required input .vtu file!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string fileName = argv[2];
    std::cout << "Parsing vtu file: " << fileName << std::endl;
    
    //read all the data from the file
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
        vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(fileName.c_str());
    reader->Update();

    // ttk::ExplicitTriangulation explicitTriangulation;
    // explicitTriangulation.createExplicitTriangulation(reader->GetOutput());

    // ttk::TestTopoRelations testPlugin;
    // testPlugin.preconditionTriangulation(&explicitTriangulation);
    // testPlugin.execute(&explicitTriangulation);

    ttk::AcTopo actopo;
    actopo.initialize(reader->GetOutput());

    ttk::TestTopoRelations testPlugin;
    testPlugin.preconditionTriangulation(&actopo);
    testPlugin.execute(&actopo);

    return 0;
}
