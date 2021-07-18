#include "fonctionnalite.h"
#include "header.h"
#include "rbf.h"

/*
def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

*/

RBF::RBF(float* x, int size_x, float *y, int size_y, float *tX, int size_tX,
         float *tY, int size_tY, int number_of_classes, int k, bool std_from_clusters = false) {
    this->x = CreateModel(x,size_x);
    this->y = CreateModel(y, size_y);
    this->tX = CreateModel(tX, size_tX);
    this->tY = CreateModel(tY, size_tY);

    this->number_of_classes = number_of_classes;
    this->k = k;
    this->std_from_clusters = std_from_clusters;
}

/*
    def convert_to_one_hot(self, x, num_of_classes):
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr
*/
Model2 *RBF::convert_to_one_hot(float *x, int sizeX, int num_of_classes) {
    Model2 *arr = model2_set_zero(sizeX, num_of_classes);
    for (int i = 0; i < sizeX; i++) {
        int c = int(x[i]);
        arr->values[i][c] = 1;
    }
    return arr;
}

/*
def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

*/
float RBF::get_distance(float* x1, int size_x1, float* x2, int size_x2) {
    float sum = 0.0f;

    for (int i = 0; i < size_x1; i += 1) {
        sum += pow(x1[i] - x2[i], 2);
    }

    return sqrt(sum);
}
/*
   def rbf_list(self, X, centroids, std_list):
       RBF_list = []
       for x in X:
           RBF_list.append([self.rbf(x, c, s) for (c, s) in zip(centroids, std_list)])
       return np.array(RBF_list)
*/
Model2 *RBF::rbf_list(Model2 *X, Model2 *centroids, Model *std_list) {
    Model2 *RBF_list = (Model2 *) (malloc(sizeof(Model2 *)));
    RBF_list->values = (float **) (malloc(sizeof(float **) * X->x));

    for (int i = 0; i < X->x; i++) {
        RBF_list->values[i] = (float *) (malloc(sizeof(float *) * std_list->size));
    }

    for (int i = 0; i < X->x; i++) {
        for (int j = 0; j < std_list->size; j += 1) {
            RBF_list->values[i][j] = this->rbf(X->values[j], X->y[j], centroids->values[j], centroids->y[j],
                                               std_list->values[j]);
        }
    }
    return RBF_list;
}

/*
    def rbf(self, x, c, s):
        distance = get_distance(x, c)
        return 1 / np.exp(-distance / s ** 2)
*/
float RBF::rbf(float *x, int sizeX, float *c, int sizeC, float s) {
    float distance = this->get_distance(x, sizeX, c, sizeC);
    return 1 / exp(-distance / pow(s, 2));
}

/*
def fit(self):

    self.centroids, self.std_list = kmeans(self.X, self.k, max_iters=1000)

    if not self.std_from_clusters:
        dMax = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
        self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

    RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)

    self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.convert_to_one_hot(self.y, self.number_of_classes)

    RBF_list_tst = self.rbf_list(self.tX, self.centroids, self.std_list)

    self.pred_ty = RBF_list_tst @ self.w

    self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])

    diff = self.pred_ty - self.ty

    print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))
*/
void fit(){

}
/*
def kmeans(X, k, max_iters):

    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    converged = False
    current_iter = 0
    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for i in range(len(centroids))]
        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)
        cluster_list = list((filter(None, cluster_list)))
        prev_centroids = centroids.copy()
        centroids = []
        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0))
        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))
        print('K-MEANS: ', int(pattern))
        converged = (pattern == 0)
        current_iter += 1
    return np.array(centroids), [np.std(x) for x in cluster_list]
 */