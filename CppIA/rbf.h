#ifndef CPPIA_RBF_H
#define CPPIA_RBF_H

/*
   def __init__(self, X, y, tX, ty, num_of_classes,
                k, std_from_clusters=True):
       self.X = X
       self.y = y

       self.tX = tX
       self.ty = ty

       self.number_of_classes = num_of_classes
       self.k = k
       self.std_from_clusters = std_from_clusters
*/

class RBF{
public:

    Model* x;
    Model* y;

    Model* tX;
    Model* tY;

    int number_of_classes;
    int k; // number of centroid
    bool std_from_clusters;

public:
    RBF(float* x, int size_x, float* y, int size_y,float* tX, int size_tX,
        float* tY, int size_tY, int number_of_classes,int k,bool std_from_clusters);
    Model2* convert_to_one_hot(float* x, int sizeX,int num_of_classes);
    float rbf(float* x, int sizeX, float* c, int sizeC, float s);
    Model2* rbf_list(Model2* X, Model2* centroids, Model* Std_list);
    void fit();
    float get_distance(float* x1, int size_x1,float* x2, int size_x2);
};
#endif //CPPIA_RBF_H
