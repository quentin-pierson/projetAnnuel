#include "header.h"
#include "fonctionnalite.h"

float* cut_float_array(float* array, int start, int end){
    int size = end-start;

    float* new_array = (float*) malloc(sizeof (float ) * size);

    for(int i = 0; i < size; i++){
        new_array[i] = array[start+i];
    }
    return new_array;
}

float* copy_array(float* array, int size){
    float* new_array = (float*) malloc(sizeof (float ) * size);

    for(int i = 0; i < size; i++){
        new_array[i] = array[i];
    }
    return new_array;
}

float* insert_array(float* array, int size,int pos, float value){
    int newSize = size+1;
    float* new_array = (float*) malloc(sizeof (float ) * (newSize));

    if(pos >= newSize){
        pos = size;
    }
    if(pos < 0){
        pos = 0;
    }

    int j = 0;
    for(int i = 0; i < newSize; i++, j++){
        if(i == pos){
            new_array[i] = value;
            j -= 1;
        }else{
            new_array[i] = array[j];
        }
    }

    free(array);

    return new_array;
}