#include <Gale.h>
#include <TestTopoRelations.h>
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <iostream>
#include <stdio.h>
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
    std::string fileName = "";
    int blockNumber = 0;
    int blockSize = 0;
    int threadNumber = 1;
    int numThreadsPerCluster = 1;

    while ((c = getopt(argc, argv, "a:b:i:p:s:t:")) != -1) {
        switch (c) {
            case 'i':
                fileName = optarg;
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
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
        vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(fileName.c_str());
    reader->Update();

    ttk::Gale gale(threadNumber, numThreadsPerCluster);
    if (blockNumber > 0 && blockSize > 0)
        gale.initialize(reader->GetOutput(), blockNumber, blockSize);
    else if (blockNumber > 0)
        gale.initialize(reader->GetOutput(), blockNumber);
    else 
        gale.initialize(reader->GetOutput());

    ttk::TestTopoRelations testPlugin;
    testPlugin.preconditionTriangulation(&gale);
    testPlugin.setThreadNumber(threadNumber);
    testPlugin.execute(&gale);

    return 0;
}
