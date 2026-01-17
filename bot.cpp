#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
using json = nlohmann::json;
using namespace std;

// Function to perform Log-Normalization
vector<float> process_snapshot(const json& bids, const json& asks) {
    vector<float> features;
    
    // Safety check
    if (bids.size() < 10 || asks.size() < 10) return features;

    float best_bid = std::stof(bids[0][0].get<string>());
    float best_ask = std::stof(asks[0][0].get<string>());
    float mid_price = (best_bid + best_ask) / 2.0f;

    // 1. Normalized Prices
    for(int i=0; i<10; i++) {
        float p = std::stof(bids[i][0].get<string>());
        features.push_back(p / mid_price);
    }
    for(int i=0; i<10; i++) {
        float p = std::stof(asks[i][0].get<string>());
        features.push_back(p / mid_price);
    }

    // 2. Normalized Volumes
    for(int i=0; i<10; i++) {
        float v = std::stof(bids[i][1].get<string>());
        features.push_back(std::log1p(v));
    }
    for(int i=0; i<10; i++) {
        float v = std::stof(asks[i][1].get<string>());
        features.push_back(std::log1p(v));
    }

    return features;
}

// Curl Write Callback
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main() {
    cout << "C++ HFT Bot Starting. " << endl;

    // 1. Setup ZeroMQ to talk to Python
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);
    cout << "Connecting to Python Brain on port 5555." << endl;
    socket.connect("tcp://localhost:5555");

    // 2. Setup Curl (HTTP)
    CURL* curl;
    CURLcode res;
    curl = curl_easy_init();
    
    if (!curl) {
        cerr << "Failed to init Curl" << endl;
        return 1;
    }

    string readBuffer;
    string url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=20";

    // Main Loop
    while(true) {
        readBuffer.clear();
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        
        // A. Fetch Data (Fast)
        res = curl_easy_perform(curl);
        
        if(res != CURLE_OK) {
            cerr << "Curl failed: " << curl_easy_strerror(res) << endl;
            continue;
        }

        // B. Parse JSON
        auto data = json::parse(readBuffer);
        
        // C. Process Features (Math)
        vector<float> features = process_snapshot(data["bids"], data["asks"]);
        
        if (features.size() != 40) {
            cout << "Bad feature size" << endl;
            continue;
        }

        // D. Send to Python (The Fast Pipe)
        zmq::message_t request(features.size() * sizeof(float));
        memcpy(request.data(), features.data(), features.size() * sizeof(float));
        socket.send(request, zmq::send_flags::none);

        // E. Receive Signal
        zmq::message_t reply;
        auto result = socket.recv(reply, zmq::recv_flags::none);
        string ans = string(static_cast<char*>(reply.data()), reply.size());

        // F. Action
        if (ans == "Buffering") {
            cout << ".";
        } else {
            // Ans format: "Action,Conf" e.g., "2,0.65"
            cout << "Model Says: " << ans << endl;
        }
    }

    curl_easy_cleanup(curl);
    return 0;
}