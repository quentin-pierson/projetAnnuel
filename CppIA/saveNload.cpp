#include "fonctionnalite.h"
#include "header.h"
#include "mlp.h"


string save_model2(Model2 *model) {
    string json = "";

    int L = model->x - 1;

    string start_char = "{ \"x\" : ";

    string middle_char = ", \"y\" : [";

    string values_char = ", \"values\" : [";

    json = start_char + to_string(model->x);
    json += middle_char;

    json = simple_array_to_string(json, model->x, model->y);
    json += "]";
    json += values_char;

    for (int i = 0; i < model->x; i++) {
        json += "[";
        // [
        for (int j = 0; j < model->y[i]; j++) {
            json += to_string(model->values[i][j]);

            if (j + 1 != model->y[i]) {
                json += ", ";
            }
        }
        json += "]";
        if (i + 1 != model->x) {
            json += ", ";
        }

    }
    json += "]}";
    return json;
}

string simple_array_to_string(string data, int x, int *y) {
    for (int i = 0; i < x; i += 1) {
        data += to_string(y[i]);

        if (i + 1 < x) {
            data += ", ";
        }
    }
    return data;
}

string simple_array_to_string(string data, int x, float *y) {
    for (int i = 0; i < x; i += 1) {
        data += to_string(y[i]);

        if (i + 1 < x) {
            data += ", ";
        }
    }
    return data;
}

string save_model3(Model3 *model) {

    string json = "";

    string start_char = "{ \"x\" : ";

    string middle_char = ", \"y\" : [";

    string values_char = ", \"values\" : [";

    json = start_char + to_string(model->x);
    json += middle_char;
    json = simple_array_to_string(json, model->x, model->y);
    json += "]" + values_char;

    for(int l=0; l< model->x ;l++){

        int imax= model->y[l-1]+1 ;
        json += "[";
        for (int i = 0; i < imax; i++){
            json += "[";
            int jmax = model->y[l] + 1;
            for (int j = 0; j < jmax; j++){
                json += to_string(model->values[l][i][j]);

                if(j +1 != jmax){
                    json += ", ";
                }
            }
            json += "]";
            if(i +1 != imax){
                json += ", ";
            }

        }
        json += "]";
        if(l +1 != model->x){
            json += ", ";
        }
    }

    json += "]}";

    return json;
}

DLLEXPORT void save_mlp_regression(MLP* mlp){
    save_mlp_model(mlp, "regression");
}

DLLEXPORT void save_mlp_classification(MLP* mlp){
    save_mlp_model(mlp, "classification");
}

void save_mlp_model(MLP *mlp, string type) {
    string data = "{ \"W\" : ";
    data += save_model3(mlp->W);
    data += ", \"d\" : ";
    data += save_model(mlp->d);
    data += ", \"X\" : " + save_model2(mlp->X);
    data += ", \"deltas\" : " + save_model2(mlp->deltas);
    data += "}";

    string name = "mlp_";

    for (int i = 0; i < mlp->W->x; i += 1) {
        name += to_string(mlp->W->y[i]) + "_";
    }

    name += type+ "_";

    save_in_json(name, data);
}

void save_in_json(string name, string data) {
    fstream file;
    time_t date = time(0);
    tm *ltm = localtime(&date);


    string filename = "../save/" + name + to_string(1900 + ltm->tm_year)
                      + to_string(1 + ltm->tm_mon) + to_string(ltm->tm_mday)
                      + to_string(ltm->tm_hour) + to_string(ltm->tm_min)
                      + to_string(ltm->tm_sec) + ".json";

    file.open(filename, fstream::out);

    if (!file.is_open()) {
        cerr << "Failed to open" << "\n";
    } else {
        file << data << endl;
    }

    file.close();
}

string save_model(Model *model) {

    string json = "";
    json += "{ \"size\" : " + to_string(model->size);

    json += ", \"values\" : [";

    json = simple_array_to_string(json, model->size, model->values);

    json += "]}";

    return json;
}

string open_json(char *filename) {
    string myText;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open" << "\n";
        return nullptr;
    }

    std::string data;

    while (getline(file, myText)) {
        data = myText;
    }

    file.close();

    return data;
}


Model *load_model(string data) {
    std::size_t found = data.find("size");

    int start = 0;
    int end = 0;

    for (int i = found; i < data.length(); i++) {
        if (data[i] == ':') {
            start = i + 1;
        }
        if (data[i] == ',') {
            end = i;
            break;
        }
    }
    int size = stoi(data.substr(start, end - start));

    Model *model = (Model *) (malloc(sizeof(Model)));

    model->size = size;

    model->values = (float *) (malloc(sizeof(float) * model->size));

    found = data.find("values");
    int cnt = 0;
    for (int i = found; i < data.length(); i++) {
        if (data[i] == '[') {
            start = i + 1;
        }
        if (data[i] == ',') {
            end = i;

            model->values[cnt] = stof(data.substr(start, end - start));
            start = i + 1;
            cnt += 1;
        }

        if (data[i] == ']') {
            end = i;
            model->values[cnt] = stof(data.substr(start, end - start));
            break;
        }

    }
    return model;
}

Model2 *load_model2(string data) {
    //values prend le double for
    std::size_t found = data.find("x");

    int start = 0;
    int end = 0;

    for (int i = found; i < data.length(); i++) {
        if (data[i] == ':') {
            start = i + 1;
        }
        if (data[i] == ',') {
            end = i;
            break;
        }
    }
    int size = stoi(data.substr(start, end - start));
    Model2 *model = (Model2 *) (malloc(sizeof(Model2)));
    model->values = (float **) (malloc(sizeof(float **) * size));

    model->x = size;
    model->y = (int *) (malloc(sizeof(int) * size));

    found = data.find("y");
    int cnt = 0;

    for (int i = found; i < data.length(); i++) {
        if (data[i] == '[') {
            start = i + 1;
        }
        if (data[i] == ',') {
            end = i;

            model->y[cnt] = stof(data.substr(start, end - start));
            start = i + 1;
            cnt += 1;
        }

        if (data[i] == ']') {
            end = i;
            model->y[cnt] = stof(data.substr(start, end - start));
            break;
        }
    }

    //values
    found = data.find("values");
    cnt = 0;
    float *X = (float *) malloc(sizeof(float) * 1);

    for (int i = found; i < data.length(); i++) {
        if (data[i] == '[') {
            start = i + 1;
        }
        if (data[i] == ',') {
            if(data[i-1] != ']'){
                end = i;

                float val = stof(data.substr(start, end - start));
                X = insert_array(X, cnt + 1, cnt, val);
                start = i + 1;
                cnt += 1;
            }
        }

        if (data[i] == ']') {
            end = i;
            float val = stof(data.substr(start, end - start));
            X = insert_array(X, cnt + 1, cnt, val);
            cnt += 1;
        }
    }


    cnt = 0;
    for (int l = 0; l < size; l++) {
        int iMax2 = model->y[l];
        model->values[l] = (float *) (malloc(sizeof(float *) * iMax2));

        for (int i = 0; i < iMax2; i++) {
            model->values[l][i] = X[cnt];
            cnt += 1;
        }
    }

    return model;
}

Model3 *load_model3(string data) {
    // x
    // -> y avec X
    // values avec y et W prend les 3 for au dessus
    std::size_t found = data.find("x");

    int start = 0;
    int end = 0;

    for (int i = found; i < data.length(); i++) {
        if (data[i] == ':') {
            start = i + 1;
        }
        if (data[i] == ',') {
            end = i;
            break;
        }
    }
    int size = stoi(data.substr(start, end - start));

    Model3 *model = (Model3 *) (malloc(sizeof(Model3)));
    model->values = (float ***) (malloc(sizeof(float ***) * size));

    model->x = size;
    model->y = (int *) (malloc(sizeof(int) * size));

    found = data.find("y");
    int cnt = 0;

    for (int i = found; i < data.length(); i++) {
        if (data[i] == '[') {
            start = i + 1;
        }
        if (data[i] == ',') {

            end = i;

            model->y[cnt] = stof(data.substr(start, end - start));
            start = i + 1;
            cnt += 1;

        }

        if (data[i] == ']') {
            end = i;
            model->y[cnt] = stof(data.substr(start, end - start));
            break;
        }

    }

    found = data.find("values");

    cnt = 0;
    float *w = (float *) malloc(sizeof(float) * 1);

    for (int i = found; i < data.length(); i++) {
        if (data[i] == '[') {
            start = i + 1;
        }
        if (data[i] == ',') {
            if(data[i-1] != ']') {
                end = i;

                float val = stof(data.substr(start, end - start));
                w = insert_array(w, cnt + 1, cnt, val);
                start = i + 1;
                cnt += 1;
            }
        }

        if (data[i] == ']') {
            if (data[i-1] != '[' &&  data[i-1] != ']') {
                end = i;
                float val = stof(data.substr(start, end - start));
                w = insert_array(w, cnt + 1, cnt, val);
                cnt += 1;
            }
        }

    }

    cnt = 0;
    for (int l = 0; l < model->x; l++) {

        int imax = model->y[l - 1] + 1;

        if (l == 0) {
            model->values[l] = (float **) (malloc(sizeof(float **) * 1));
            continue;
        } else {
            model->values[l] = (float **) (malloc(sizeof(float **) * imax));
        }

        for (int i = 0; i < imax; i++) {
            int jmax = model->y[l] + 1;
            model->values[l][i] = (float *) (malloc(sizeof(float *) * jmax));
            for (int j = 0; j < jmax; j++) {
                model->values[l][i][j] = w[cnt];
                cnt += 1;
            }
        }
    }

    return model;
}

string json_find_value(string data, string value) {
    std::size_t found = data.find(value);

    int start = 0;
    int end = 0;
    int count_bracket = 0;

    for (int i = found; i < data.length(); i++) {
        if (data[i] == '{') {
            count_bracket += 1;
            if (start == 0) start = i + 1;
        }
        if (data[i] == '}') {
            count_bracket -= 1;
            if (count_bracket == 0) {
                end = i;
                break;
            }
        }
    }
    return data.substr(start, end - start);

}

DLLEXPORT MLP* load_mlp_model(char *filename) {
    string data = open_json(filename);
    string data_W = json_find_value(data, "W");
    string data_d = json_find_value(data, "d");
    string data_X = json_find_value(data, "X");
    string data_deltas = json_find_value(data, "deltas");


    Model3 *W = load_model3(data_W);
    Model *d = load_model(data_d);
    Model2 *X = load_model2(data_X);
    Model2 *deltas = load_model2(data_deltas);

    MLP *mlp = new MLP(W, d, X, deltas);
    return mlp;
}

