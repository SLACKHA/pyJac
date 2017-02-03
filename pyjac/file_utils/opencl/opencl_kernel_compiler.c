#include "ocl_errorcheck.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>
#include <assert.h>

#define MAX_DEVICE (100)

int main(int argc, char* argv[])
{
    cl_platform_id platform_id[10];
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;

    cl_int return_code;

    //check args
    // argv[1] == filename
    // argv[2] == platform id
    // argv[3] == output file name
    // argv[4] == build options (optional)
    assert(argc >= 4);

    char* filename = (char*)malloc(strlen(argv[1]) * sizeof(char));
    sprintf(filename, argv[1]);

    char* platform_check = (char*)malloc(strlen(argv[2]) * sizeof(char));
    sprintf(platform_check, argv[2]);

    char* outname = (char*)malloc(strlen(argv[3]) * sizeof(char));
    sprintf(outname, argv[3]);

    char* build_options = NULL;
    if (argc >= 5)
    {
        size_t len = 0;
        //all further options are build options
        //first find total len
        for(int argi = 4; argi < argc; ++argi)
        {
            len += strlen(argv[argi]);
            if (argi != argc - 1)
            {
                len += 1; //for space separator
            }
        }
        build_options = (char*)malloc(len * sizeof(char));
        //next place into build options
        size_t offset = 0;
        for(int argi = 4; argi < argc; ++argi)
        {
            sprintf(&build_options[offset], argv[argi]);
            offset += strlen(argv[argi]);
            if (argi != argc - 1)
            {
                build_options[offset] = ' ';
                offset += 1;
            }
        }
    }

    FILE *fp;
    char *source_str;

    /* Load kernel source code */
    fp = fopen(filename, "r");
    if (!fp) {
        exit(-1);
    }
    //find file size
    fseek(fp, 0L, SEEK_END);
    size_t source_size = ftell(fp);
    rewind(fp);

    //read file
    source_str = (char*)malloc(source_size);
    assert(fread(source_str, 1, source_size, fp) == source_size);
    fclose(fp);

    /* Get platform/device information */
    check_err(clGetPlatformIDs(10, platform_id, &ret_num_platforms));
    cl_platform_id pid = NULL;
    for (int i = 0; i < ret_num_platforms; ++i)
    {
        //check if intel
        char pvendor[500];
        size_t psize = 500 * sizeof(char);
        check_err(clGetPlatformInfo(platform_id[i], CL_PLATFORM_VENDOR, psize, pvendor, NULL));
        if(strstr(pvendor, platform_check) != NULL)
        {
            pid = platform_id[i];
            break;
        }
    }


    //get the device to compile for
    check_err(clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices));

    //create context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &return_code);
    check_err(return_code);

    //create queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &return_code);
    check_err(return_code);

    /* Create Kernel program from the read in source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &return_code);
    check_err(return_code);

    /* Build Kernel Program */
    return_code = clBuildProgram(program, 1, &device_id, build_options, NULL, NULL);
    if (return_code != CL_SUCCESS)
    {
          printf("OpenCL failed to build the program...\n");

          size_t len;
          char *buffer;
          check_err(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(char*), NULL, &len));
          buffer = calloc(len, sizeof(char));
          check_err(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len * sizeof(char), buffer, NULL));
          printf("%s\n", buffer);
          free(buffer);

          clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(char*), NULL, &len);
          buffer = calloc(len, sizeof(char));
          clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, len * sizeof(char), buffer, NULL);
          printf("%s\n", buffer);
          free(buffer);

          check_err(return_code);
    }

    // Get compiled binary from runtime

    //get # of programs
    size_t num_prog;
    check_err(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), NULL, &num_prog));
    num_prog /= sizeof(size_t);
    assert(num_prog == 1);

    //get program size
    size_t bin_size;
    check_err(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, num_prog * sizeof(size_t), &bin_size, NULL));

    //get binary
    unsigned char *binary = malloc(bin_size);
    check_err(clGetProgramInfo(program, CL_PROGRAM_BINARIES, num_prog * sizeof(unsigned char*), &binary, NULL));

    // Then write binary to file
    fp = fopen(outname, "wb");
    if (!fp) {
        exit(-1);
    }
    assert(fwrite(binary, bin_size, 1, fp) == 1);

    free(filename);
    free(platform_check);
    free(outname);
    free(build_options);
    free(source_str);
    free(binary);
}