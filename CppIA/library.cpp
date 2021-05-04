#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

DLLEXPORT int toto() {
    return 42;
}

DLLEXPORT int array_sum(int arr[], int arr_size){
    int total = 0;
    for (int i=0;i<arr_size;i++){
        total += arr[i];
    }
    return total;
}