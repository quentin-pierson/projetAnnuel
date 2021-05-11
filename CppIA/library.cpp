#include "header.h"

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

