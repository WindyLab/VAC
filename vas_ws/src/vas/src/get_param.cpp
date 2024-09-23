#include "get_param.h"

void ParamManager::loadParam() {
    std::string param_path = "../cali_file/params.json";
    std::ifstream file(param_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open ../cali_file/params.json");
    }

    nlohmann::json json_f;// = nlohmann::json::parse(file, nullptr, true);
    file >> json_f;

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [key, value] : json_f.items()) {
        params_[key] = value.dump();
    }
}

std::string ParamManager::getParam(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    return params_[key];
}


// int main(int argc, char** argv) {
//     ParamManager::getInstance().loadParam();

//     std::string k = ParamManager::getInstance().getParam("k");
//     std::string k_sep = ParamManager::getInstance().getParam("k_sep");
//     std::string v_max = ParamManager::getInstance().getParam("v_max");

//     std::cout << "k: " << std::stof(k) << "\n";
//     std::cout << "k_sep: " << std::stof(k_sep)  << "\n";
//     std::cout << "v_max: " << std::stof(v_max)  << "\n";

//     return 0;
// }
