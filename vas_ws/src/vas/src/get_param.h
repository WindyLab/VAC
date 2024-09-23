#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <mutex>
#include "json.hpp"

class ParamManager {
public:
    static ParamManager& getInstance() {
        static ParamManager instance;
        return instance;
    }

 ParamManager(const ParamManager&) = delete;
 ParamManager& operator=(const ParamManager&) = delete;
    void loadParam(); 
    std::string getParam(const std::string& key); 

private:
 ParamManager() {}
    std::mutex mutex_;
    std::unordered_map<std::string, std::string> params_;
};