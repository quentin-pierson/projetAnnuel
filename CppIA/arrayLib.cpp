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

void printW(Model3* W){
    std::cout << "[ ";
    for(int l=0; l< W->x ;l++){ //
        int imax= W->y[l-1]+1 ;
        std::cout << "[ ";
        if(l==0){
            std::cout << "], ";
            continue;
        }
        for (int i = 0; i < imax; i++){
            std::cout << "[ ";
            int jmax = W->y[l] + 1;
            for (int j = 0; j < jmax; j++){
                std::cout << W->values[l][i][j] << ", ";
            }
            std::cout << "], ";
        }
        std::cout << "], "<< "\n";
    }
    std::cout << "] "<< "\n";
}

void printX(Model2* X){
    std::cout << "[ ";
    for(int l=0; l< X->x ;l++){ //
        int iMax = X->y[l];
        std::cout << "[ ";

        for (int i = 0; i < iMax; i++){
            std::cout << X->values[l][i] << ", ";
        }
        std::cout << "], "<< "\n";
    }
    std::cout << "] "<< "\n";
}

Model2* CreateModel2(int size){
    Model2* model2 = (Model2*) (malloc( sizeof(Model2)));
    model2->values = (float **) (malloc(sizeof(float **) * size));

    model2->x = size;
    model2->y = (int *) (malloc(sizeof(int) * size));

    return model2;
}

float* TakeLast(Model2* model){
    int xSize = model->x;
    float * tab = (float *) (malloc(sizeof(float *) * model->y[xSize-1] -1));

    for (int i = 1; i < model->y[xSize-1]; i++){
        tab[i-1] = model->values[xSize-1][i];
    }
    return tab;
}